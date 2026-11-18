"""Warm starting, decomposed under one perturbation model for every solver.

Three effects are routinely reported as one number called "the warm-start
speed-up". This script separates them on a single sequence of problems:

    cold  : a fresh solver object every step (setup + factorize + solve)
    reset : one persistent object, data updated, previous iterate DISCARDED
    warm  : one persistent object, data updated, previous iterate KEPT

cold/reset is setup and factorization reuse; reset/warm is the warm start
itself. A cold/warm ratio -- which is what most reported speed-ups are --
bundles the two and attributes the total to the warm start.

Getting the reset leg right is the whole difficulty, because it has to mean
something different for each solver's API:

  OSQP    the `warm_start` setting (`warm_starting` from version 1.0) is
          turned off on the persistent object, so each solve restarts from
          zero while keeping the factorization.
  ProxQP  the shipped initial guess is recomputed every step; the warm leg
          switches `initial_guess` to WARM_START_WITH_PREVIOUS_RESULT, which
          must be set AFTER the first solve or the workspace is unpopulated.
  DAQP    `update()` retains the working set, so there is no reset by
          default. Passing a zeroed `sense` array clears the active-set flags
          while keeping the allocated workspace and the LDP representation.
  PIQP    an interior-point method has no iterate to carry over and the
          solver exposes no warm-start setting, so the warm leg DOES NOT
          EXIST. It is reported as not available, not as a measured 1.00x.

The perturbation model is a closed loop. The sequence is generated once, by
solving each QP with a reference solver, applying the first control block of
its solution to the single-rigid-body dynamics, propagating the state and
shifting the reference. The resulting linear terms are recorded and replayed
to every solver, so the loop is genuinely closed and every solver still sees
exactly the same sequence; closing the loop separately per solver would give
each a different sequence and the comparison would no longer rest on one
perturbation model.

An undisturbed closed loop is a degenerate benchmark: the controller tracks
the reference, the state settles, and consecutive problems become nearly
identical (a median step of 0.004% of ||q||), which pushes every effect below
the harness's own drift floor. How far apart consecutive problems actually
are is a property of the disturbance a robot is rejecting, not of the solver,
and the warm-start benefit depends strongly on it. So the loop is driven by a
per-step velocity disturbance of controlled magnitude and the whole
decomposition is reported as a function of the resulting median step size.
Reporting a single warm-start number without the step size it was measured at
states a solver property that does not exist.

Every leg checks the solver's objective value against the cold leg. A leg
whose objective disagrees is reported as a disagreement rather than as a
speed-up, because a solver that stops early looks fast.

The whole experiment is repeated (--reps) and the per-leg median of the
repeats is reported, together with the largest max/min spread across repeats.
That spread is the harness's own drift floor: an effect smaller than it
cannot be distinguished from run-to-run variation.

    python warmstart_controlled.py --reps 3 --out warmstart_controlled.json
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from ctypes import c_int

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems  # noqa: E402

NU = 12          # 4 contacts x 3 force components
NX = 13          # SRBD state, gravity row included
OBJ_TOL = 1e-6   # relative objective agreement required between legs


def _condensing(horizon, dt=0.03):
    """S_x, S_u and the stage weights, matching problems.srbd_mpc."""
    foot = np.array([[0.19, 0.11, -0.30], [0.19, -0.11, -0.30],
                     [-0.19, 0.11, -0.30], [-0.19, -0.11, -0.30]])
    A_d, B_d = problems._srbd_matrices(12.0, np.array([0.07, 0.26, 0.24]),
                                       foot, dt)
    N = horizon
    Sx = np.zeros((N * NX, NX))
    Su = np.zeros((N * NX, N * NU))
    for k in range(N):
        Sx[k * NX:(k + 1) * NX, :] = np.linalg.matrix_power(A_d, k + 1)
        for j in range(k + 1):
            Su[k * NX:(k + 1) * NX, j * NU:(j + 1) * NU] = \
                np.linalg.matrix_power(A_d, k - j) @ B_d
    q_state = np.array([20., 20., 5., 10., 10., 50.,
                        .5, .5, .2, 1., 1., 1., 0.])
    Qbar = np.kron(np.eye(N), np.diag(q_state) + 1e-6 * np.eye(NX))
    return A_d, B_d, Sx, Su, Qbar


def closed_loop_sequence(steps=100, horizon=10, seed=0, dt=0.03,
                         sigma=0.0):
    """A receding-horizon sequence produced by closing the loop.

    `sigma` is the standard deviation of a zero-mean velocity disturbance
    applied to the state after each propagation, in SI units (rad/s on the
    angular rates, m/s on the linear ones). It sets how far apart consecutive
    problems are; the measured step size is returned rather than assumed.

    Returns (P, G, h, [q_0 .. q_{steps-1}], info). The reference solver is
    DAQP through its native API; `info` records its exit flags, so a sequence
    generated from a failed solve cannot pass unnoticed.
    """
    import daqp

    d0 = problems.srbd_mpc(horizon=horizon, form="condensed", seed=seed)
    P, G, h = d0["P_dense"], d0["G_dense"], d0["h"]
    A_d, B_d, Sx, Su, Qbar = _condensing(horizon, dt)
    N, ni = horizon, G.shape[0]

    Pc = np.ascontiguousarray(P)
    Gc = np.ascontiguousarray(G)
    hc = np.ascontiguousarray(h)
    blc = np.full(ni, -1e30)
    sense = np.zeros(ni, dtype=c_int)

    rng = np.random.default_rng(seed)
    x = np.zeros(NX)
    x[12] = -9.81
    x[3:6] = np.array([0.0, 0.0, 0.30])
    x[0:3] = rng.normal(0.0, 0.03, 3)
    x[9:12] = rng.normal(0.0, 0.05, 3)

    def linear_term(xc, t):
        xref = np.tile(xc, (N, 1))
        for k in range(N):
            xref[k, 0:3] = 0.0
            xref[k, 3:6] = np.array([0.4 * (t + dt * (k + 1)), 0.0, 0.30])
            xref[k, 9:12] = np.array([0.4, 0.0, 0.0])
            xref[k, 12] = -9.81
        return Su.T @ Qbar @ (Sx @ xc - xref.reshape(-1))

    qs, flags, t = [], [], 0.0
    for _ in range(steps):
        q = np.ascontiguousarray(linear_term(x, t))
        qs.append(q)
        m = daqp.Model()
        m.setup(Pc, q, Gc, hc, blc, sense.copy())
        xu, fval, exitflag, _ = m.solve()
        flags.append(int(exitflag))
        u0 = np.asarray(xu[:NU], dtype=float)   # close the loop
        x = A_d @ x + B_d @ u0
        if sigma > 0.0:                         # the disturbance being rejected
            x[0:3] += rng.normal(0.0, sigma, 3)
            x[9:12] += rng.normal(0.0, sigma, 3)
        t += dt

    rel = [float(np.linalg.norm(qs[i + 1] - qs[i]) / np.linalg.norm(qs[i]))
           for i in range(len(qs) - 1)]
    assert np.all(np.isfinite(P)) and all(np.all(np.isfinite(q)) for q in qs), \
        "non-finite problem data"
    info = dict(sigma=float(sigma),
                median_rel_step=float(np.median(rel)),
                max_rel_step=float(np.max(rel)),
                min_rel_step=float(np.min(rel)),
                reference_solver="daqp (native API)",
                reference_exitflags=sorted(set(flags)),
                closed_loop=True)
    return P, G, h, qs, info


def med_ms(ts):
    return statistics.median(ts) * 1e3


def obj(P, q, x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ (P @ x) + q @ x)


def agree(a, b):
    if a is None or b is None:
        return None
    return max(abs(u - v) / max(1.0, abs(u)) for u, v in zip(a, b))


def _daqp_iter(info):
    """DAQP reports its iteration count in the info struct, under one of a
    few names depending on the binding version; None if unavailable."""
    for k in ("iter", "iterations", "nodes"):
        v = getattr(info, k, None) if not isinstance(info, dict) \
            else info.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def exact_reference(P, G, h, qs):
    """Objective at the exact optimum of every problem in the sequence.

    DAQP is a dual active-set method and terminates at an exact vertex of the
    active set, so it serves as the reference against which every leg's
    objective is scored. The reference is independent of the solver under
    test for three of the four, and for DAQP itself the comparison is between
    its own legs.
    """
    import daqp
    ni = G.shape[0]
    Pc = np.ascontiguousarray(P)
    Gc = np.ascontiguousarray(G)
    hc = np.ascontiguousarray(h)
    blc = np.full(ni, -1e30)
    out = []
    for q in qs:
        m = daqp.Model()
        m.setup(Pc, np.ascontiguousarray(q), Gc, hc, blc,
                np.zeros(ni, dtype=c_int))
        xu, fval, flag, _ = m.solve()
        out.append(obj(P, q, xu))
    return out


def legs_osqp(P, G, h, qs):
    import osqp
    import scipy.sparse as sp
    Pm, Am = sp.csc_matrix(P), sp.csc_matrix(G)
    lo = np.full(len(h), -np.inf)
    used = {}

    def setup(q, warm):
        for kw in ("warm_start", "warm_starting"):
            m = osqp.OSQP()
            try:
                m.setup(P=Pm, q=q, A=Am, l=lo, u=h, verbose=False,
                        **{kw: warm})
                used["kw"] = kw
                return m
            except TypeError:
                continue
        raise RuntimeError("no warm-start setting accepted by this OSQP")

    cold_t, cold_o, cold_i = [], [], []
    for q in qs:
        t0 = time.perf_counter()
        m = setup(q, True)
        r = m.solve()
        cold_t.append(time.perf_counter() - t0)
        cold_o.append(obj(P, q, r.x))
        cold_i.append(int(r.info.iter))

    out = {"cold": (cold_t, cold_o, cold_i)}
    for name, warm in (("reset", False), ("warm", True)):
        m = setup(qs[0], warm)
        m.solve()
        ts, os_, it = [], [], []
        for q in qs:
            t0 = time.perf_counter()
            m.update(q=q)
            r = m.solve()
            ts.append(time.perf_counter() - t0)
            os_.append(obj(P, q, r.x))
            it.append(int(r.info.iter))
        out[name] = (ts, os_, it)
    return out, (f"persistent object; `{used['kw']}` toggled for the reset "
                 f"leg. Default `check_termination` is 25, so no solve can "
                 f"report fewer iterations than that")


def legs_proxqp(P, G, h, qs):
    import proxsuite
    n, ni = P.shape[0], G.shape[0]
    IG = proxsuite.proxqp.InitialGuess

    def fresh(q):
        qp = proxsuite.proxqp.dense.QP(n, 0, ni)
        qp.init(P, q, None, None, G, None, h)
        qp.settings.verbose = False
        # Non-default and necessary: at the shipped threshold ProxQP declares
        # these problems primal infeasible after one iteration (the paper's
        # infeasibility finding). Disclosed in the note below.
        qp.settings.eps_primal_inf = 1e-9
        return qp

    cold_t, cold_o, cold_i = [], [], []
    for q in qs:
        t0 = time.perf_counter()
        qp = fresh(q)
        qp.solve()
        cold_t.append(time.perf_counter() - t0)
        cold_o.append(obj(P, q, qp.results.x))
        cold_i.append(int(qp.results.info.iter))
    out = {"cold": (cold_t, cold_o, cold_i)}

    for leg, use_warm in (("reset", False), ("warm", True)):
        qp = fresh(qs[0]); qp.solve()
        if use_warm:
            qp.settings.initial_guess = IG.WARM_START_WITH_PREVIOUS_RESULT
        ts, os_, it = [], [], []
        for q in qs:
            t0 = time.perf_counter()
            qp.update(g=q)
            qp.solve()
            ts.append(time.perf_counter() - t0)
            os_.append(obj(P, q, qp.results.x))
            it.append(int(qp.results.info.iter))
        out[leg] = (ts, os_, it)
    return out, ("WARM_START_WITH_PREVIOUS_RESULT set after the first solve; "
                 "eps_primal_inf=1e-9 (non-default) on every leg")


def legs_piqp(P, G, h, qs):
    """PIQP: cold and reset only.

    The solver exposes no warm-start setting and an interior-point method has
    no iterate to carry over, so the warm leg does not exist. Reporting it as
    1.00x would present the absence of a facility as a measurement.
    """
    import piqp
    n, ni = P.shape[0], G.shape[0]
    Pf, Gf = np.asfortranarray(P), np.asfortranarray(G)
    hl = np.full(ni, -np.inf)
    xinf = np.full(n, np.inf)

    def setup(q):
        s = piqp.DenseSolver()
        s.settings.verbose = False
        s.setup(Pf, q, None, None, Gf, hl, h, -xinf, xinf)
        return s

    cold_t, cold_o, cold_i = [], [], []
    for q in qs:
        t0 = time.perf_counter()
        s = setup(q)
        s.solve()
        cold_t.append(time.perf_counter() - t0)
        cold_o.append(obj(P, q, s.result.x))
        cold_i.append(int(s.result.info.iter))

    s = setup(qs[0]); s.solve()
    ts, os_, it = [], [], []
    for q in qs:
        t0 = time.perf_counter()
        s.update(c=q)
        s.solve()
        ts.append(time.perf_counter() - t0)
        os_.append(obj(P, q, s.result.x))
        it.append(int(s.result.info.iter))
    return ({"cold": (cold_t, cold_o, cold_i), "reset": (ts, os_, it),
             "warm": None},
            "no warm-start facility (interior point); warm leg not available")


def legs_daqp(P, G, h, qs):
    """DAQP: the reset leg clears the working set via a zeroed `sense`.

    `update()` retains the active-set flags, so without this the reset and
    warm legs are the same measurement and cold/warm gets attributed to the
    warm start.
    """
    import daqp
    ni = G.shape[0]
    Pc = np.ascontiguousarray(P, dtype=np.float64)
    Gc = np.ascontiguousarray(G, dtype=np.float64)
    hc = np.ascontiguousarray(h, dtype=np.float64)
    blc = np.full(ni, -1e30)
    zs = np.zeros(ni, dtype=c_int)
    qs = [np.ascontiguousarray(q, dtype=np.float64) for q in qs]

    cold_t, cold_o, cold_i = [], [], []
    for q in qs:
        t0 = time.perf_counter()
        m = daqp.Model()                       # inside the timer: setup cost
        m.setup(Pc, q, Gc, hc, blc, zs.copy())
        xu, fval, flag, _ = m.solve()
        cold_t.append(time.perf_counter() - t0)
        cold_o.append(obj(P, q, xu))
        cold_i.append(_daqp_iter(_))

    out = {"cold": (cold_t, cold_o, cold_i)}
    for name, clear in (("reset", True), ("warm", False)):
        m = daqp.Model()
        m.setup(Pc, qs[0], Gc, hc, blc, zs.copy())
        m.solve()
        ts, os_, it = [], [], []
        for q in qs:
            t0 = time.perf_counter()
            if clear:
                m.update(None, q, None, None, None, np.zeros(ni, dtype=c_int))
            else:
                m.update(None, q, None, None, None)
            xu, fval, flag, inf = m.solve()
            ts.append(time.perf_counter() - t0)
            os_.append(obj(P, q, xu))
            it.append(_daqp_iter(inf))
        out[name] = (ts, os_, it)
    return out, "working set cleared by a zeroed `sense` on the reset leg"


RUNNERS = {"OSQP": legs_osqp, "ProxQP": legs_proxqp,
           "PIQP": legs_piqp, "DAQP": legs_daqp}

GENERIC_NAME = {"OSQP": "osqp", "ProxQP": "proxqp",
                "PIQP": "piqp", "DAQP": "daqp"}


def legs_generic(P, G, h, qs, solver):
    """The generic `qpsolvers` path, on the same sequence as the legs above.

    An earlier version of this comparison drew an independent problem at
    every step, so a point carried across steps had nothing in common with the
    next problem and "no benefit" was a tautology rather than a finding. Here
    the generic interface sees the same closed-loop sequence as the legs
    above, so the comparison is about the interface and not the sequence.
    `n_guessed` records how often `initvals` was actually populated: a step
    whose solve does not report `found` supplies no point.
    """
    from qpsolvers import Problem, solve_problem
    probs = [Problem(P, q, G, h, None, None) for q in qs]

    cold = []
    for pr in probs:
        t0 = time.perf_counter()
        solve_problem(pr, solver=solver)
        cold.append(time.perf_counter() - t0)

    warm, prev, n_guessed = [], None, 0
    for pr in probs:
        t0 = time.perf_counter()
        if prev is None:
            sol = solve_problem(pr, solver=solver)
        else:
            sol = solve_problem(pr, solver=solver, initvals=prev)
            n_guessed += 1
        warm.append(time.perf_counter() - t0)
        if getattr(sol, "found", False) and sol.x is not None:
            prev = sol.x
    return med_ms(cold), med_ms(warm), n_guessed


def osqp_check_termination_probe(P, G, h, qs, intervals=(25, 1)):
    """Why OSQP's warm start measures nothing at its default settings.

    OSQP checks its termination criteria every `check_termination` iterations,
    25 by default, so no solve can report fewer than 25. If a warm-started
    solve would converge in two, the saving is invisible in both the
    iteration count and the wall time. This probe repeats the reset/warm
    comparison at the default interval and at 1, which separates the
    algorithm's warm start from the reporting granularity that hides it.
    """
    import osqp
    import scipy.sparse as sp
    Pm, Am = sp.csc_matrix(P), sp.csc_matrix(G)
    lo = np.full(len(h), -np.inf)
    out = {}
    for ct in intervals:
        legs = {}
        for leg, warm in (("reset", False), ("warm", True)):
            m = osqp.OSQP()
            for kw in ("warm_start", "warm_starting"):
                try:
                    m.setup(P=Pm, q=qs[0], A=Am, l=lo, u=h, verbose=False,
                            check_termination=ct, **{kw: warm})
                    break
                except TypeError:
                    m = osqp.OSQP()
            m.solve()
            ts, its = [], []
            for q in qs:
                t0 = time.perf_counter()
                m.update(q=q)
                r = m.solve()
                ts.append(time.perf_counter() - t0)
                its.append(int(r.info.iter))
            legs[leg] = dict(median_ms=med_ms(ts),
                             median_iters=statistics.median(its))
        out[str(ct)] = dict(
            reset=legs["reset"], warm=legs["warm"],
            warm_start=legs["reset"]["median_ms"] / legs["warm"]["median_ms"],
            iter_ratio=(legs["reset"]["median_iters"]
                        / max(1e-9, legs["warm"]["median_iters"])))
    return out


def spread(vals):
    vals = [v for v in vals if v and v > 0]
    return (max(vals) / min(vals)) if len(vals) > 1 else float("nan")


def environment():
    env = dict(platform=platform.platform(), machine=platform.machine(),
               python=platform.python_version(), numpy=np.__version__)
    for mod in ("osqp", "proxsuite", "piqp", "daqp", "qpsolvers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def measure(P, G, h, qs, reps, ref):
    """The three legs plus the generic path, for every solver, `reps` times.

    `ref` is the exact objective of every problem in the sequence. Each leg's
    objective is scored against it, so a leg that is fast because it stopped
    early is visible as an accuracy gap rather than as a speed-up.
    """
    acc = {n: dict(cold=[], reset=[], warm=[], gen_cold=[], gen_warm=[],
                   n_guessed=None, note=None, objs={}, iters={}, error=None)
           for n in RUNNERS}
    for _ in range(reps):
        for name, fn in RUNNERS.items():
            r = acc[name]
            try:
                legs, note = fn(P, G, h, qs)
                r["note"] = note
                for leg in ("cold", "reset", "warm"):
                    if legs.get(leg) is None:
                        continue
                    ts, objs, iters = legs[leg]
                    r[leg].append(med_ms(ts))
                    r["objs"][leg] = objs
                    if any(i is not None for i in iters):
                        r["iters"][leg] = statistics.median(
                            [i for i in iters if i is not None])
            except Exception as exc:
                r["error"] = f"{type(exc).__name__}: {exc}"
            try:
                gc, gw, ng = legs_generic(P, G, h, qs, GENERIC_NAME[name])
                r["gen_cold"].append(gc)
                r["gen_warm"].append(gw)
                r["n_guessed"] = ng
            except Exception as exc:
                if "gen_error" not in r:
                    r["gen_error"] = f"{type(exc).__name__}: {exc}"

    results, floors = {}, []
    for name, r in acc.items():
        if r["error"] and not r["cold"]:
            results[name] = dict(error=r["error"])
            continue
        cm = statistics.median(r["cold"])
        rm = statistics.median(r["reset"]) if r["reset"] else None
        wm = statistics.median(r["warm"]) if r["warm"] else None
        rec = dict(reps=len(r["cold"]), note=r["note"], cold_ms=cm,
                   reset_ms=rm, warm_ms=wm,
                   setup_reuse=(cm / rm) if rm else None,
                   warm_start=(rm / wm) if (rm and wm) else None,
                   total=(cm / wm) if wm else None,
                   warm_leg_available=wm is not None)
        # accuracy of each leg against the exact optimum, and consistency
        # between the legs of one solver
        rec["obj_gap_vs_exact"] = {
            leg: agree(ref, r["objs"].get(leg))
            for leg in ("cold", "reset", "warm")
            if r["objs"].get(leg) is not None}
        base = r["objs"].get("cold")
        rec["leg_consistency"] = {
            leg: agree(base, r["objs"].get(leg))
            for leg in ("reset", "warm") if r["objs"].get(leg) is not None}
        # a leg is only comparable if it is no less accurate than the cold
        # leg by more than an order of magnitude
        gaps = rec["obj_gap_vs_exact"]
        cg = gaps.get("cold")
        rec["obj_ok"] = (all(v <= max(10 * cg, OBJ_TOL)
                             for v in gaps.values())
                         if cg is not None else None)
        rec["median_iters"] = dict(r["iters"])
        d = [x for x in (spread(r["cold"]), spread(r["reset"]),
                         spread(r["warm"])) if x == x]
        rec["drift"] = max(d) if d else None
        if d:
            floors.append(max(d))
        if r["gen_cold"]:
            gcm = statistics.median(r["gen_cold"])
            gwm = statistics.median(r["gen_warm"])
            rec.update(generic_cold_ms=gcm, generic_warm_ms=gwm,
                       generic_speedup=gcm / gwm,
                       generic_steps_with_guess=r["n_guessed"])
        elif r.get("gen_error"):
            rec["generic_error"] = r["gen_error"]
        results[name] = rec
    return results, (max(floors) if floors else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sigmas", default="0,2e-5,1e-4,5e-4,2e-3",
                    help="velocity-disturbance standard deviations to sweep")
    ap.add_argument("--out", default="warmstart_controlled.json")
    a = ap.parse_args()

    sigmas = [float(x) for x in a.sigmas.split(",")]
    out = dict(steps=a.steps, horizon=a.horizon, reps=a.reps,
               env=environment(), obj_tol=OBJ_TOL, sweep=[])
    floors = []

    for sigma in sigmas:
        P, G, h, qs, info = closed_loop_sequence(a.steps, a.horizon,
                                                 sigma=sigma)
        step = info["median_rel_step"] * 100
        print(f"\n=== sigma={sigma:g}  median step {step:.3f}% of ||q||  "
              f"(range {info['min_rel_step'] * 100:.3f}-"
              f"{info['max_rel_step'] * 100:.3f}%), reference exitflags "
              f"{info['reference_exitflags']}")
        ref = exact_reference(P, G, h, qs)
        res, floor = measure(P, G, h, qs, a.reps, ref)
        probe = osqp_check_termination_probe(P, G, h, qs)
        if floor:
            floors.append(floor)
        out["sweep"].append(dict(perturbation=info, drift_floor=floor,
                                 results=res,
                                 osqp_check_termination=probe))
        print(f"{'solver':8s} {'cold':>8s} {'reset':>8s} {'warm':>8s} "
              f"{'setup':>8s} {'warmst':>8s} {'drift':>6s} "
              f"{'iters(c/r/w)':>14s} {'gap':>9s}")
        for name, rec in res.items():
            if "error" in rec:
                print(f"{name:8s} ERROR {rec['error']}")
                continue

            def fx(v):
                return ("%7.2fx" % v) if v else "    n/a"

            def fm(v):
                return ("%7.3f" % v) if v else "    n/a"
            it = rec.get("median_iters", {})
            itxt = "/".join(("%g" % it[k]) if k in it else "-"
                            for k in ("cold", "reset", "warm"))
            gap = rec.get("obj_gap_vs_exact", {}).get("cold")
            gtxt = ("%9.1e" % gap) if gap is not None else "        -"
            print(f"{name:8s} {fm(rec['cold_ms'])}  {fm(rec['reset_ms'])}  "
                  f"{fm(rec['warm_ms'])}  {fx(rec['setup_reuse'])} "
                  f"{fx(rec['warm_start'])} "
                  f"{(('%5.2fx' % rec['drift']) if rec['drift'] else '  n/a')} "
                  f"{itxt:>14s} {gtxt:>9s}"
                  + ("" if rec.get("obj_ok") in (True, None)
                     else "  LEG LESS ACCURATE"))

        for ct, r in probe.items():
            print(f"{'':8s} OSQP check_termination={ct:>2s}: reset "
                  f"{r['reset']['median_iters']:.0f} it "
                  f"{r['reset']['median_ms']:.3f} ms, warm "
                  f"{r['warm']['median_iters']:.0f} it "
                  f"{r['warm']['median_ms']:.3f} ms -> warm start "
                  f"{r['warm_start']:.2f}x")

    out["drift_floor"] = max(floors) if floors else None
    if out["drift_floor"]:
        print(f"\nharness drift floor over the whole sweep: "
              f"{out['drift_floor']:.2f}x")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

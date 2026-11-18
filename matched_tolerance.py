"""Speed at a COMMON tolerance, and suboptimality against a certified optimum.

The main benchmark runs every solver at its own defaults, which is what a
practitioner meets but is not a like-for-like speed comparison: the solvers
stop at tolerances that differ by ten orders of magnitude. This script repeats
the comparison with eps_abs/eps_rel pinned to a common target wherever the
wrapper exposes them, and reports relative suboptimality
|f(x)-f*|/max(1,|f*|) against a high-accuracy reference -- a measure no
algorithm family wins by construction, unlike a one-sided constraint
violation, which an interior-point method scores zero on for free.
"""
import argparse, json, os, statistics, sys, time
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform
import problems, qpsolvers
from qpsolvers import Problem, solve_problem

# "A common epsilon" is not one setting: each solver spells its stopping
# rule differently and the mapping is stated here rather than implied. Every
# solver that exposes an absolute tolerance gets 1e-9; iteration caps are
# raised only where the default cap would bind before the tolerance does.
EPS = 1e-9
TOL = {"osqp": dict(eps_abs=EPS, eps_rel=EPS, max_iter=100000),
       "scs": dict(eps_abs=EPS, eps_rel=EPS),
       "qpalm": dict(eps_abs=EPS, eps_rel=EPS, max_iter=100000),
       "proxqp": dict(eps_abs=EPS, eps_rel=EPS),
       "piqp": dict(eps_abs=EPS, eps_rel=EPS),
       "clarabel": dict(tol_gap_abs=EPS, tol_gap_rel=EPS, tol_feas=EPS),
       "ecos": dict(abstol=EPS, reltol=EPS, feastol=EPS),
       "cvxopt": dict(abstol=EPS, reltol=EPS),
       "highs": dict(primal_feasibility_tolerance=EPS,
                     dual_feasibility_tolerance=EPS),
       "daqp": dict(primal_tol=EPS)}
# quadprog is a dual active-set method with no tolerance to set: it
# terminates at an exact vertex, so its two columns are identical by design.

REF_SOLVERS = ("clarabel", "piqp", "daqp", "quadprog")


SOLVER_MODULES = ("osqp", "proxsuite", "piqp", "daqp", "scs", "clarabel",
                  "qpalm", "ecos", "cvxopt", "quadprog", "highspy")


def solver_versions():
    """Version of every solver backend present, for the released JSON."""
    import importlib
    out = {}
    for m in SOLVER_MODULES:
        try:
            out[m] = getattr(importlib.import_module(m), "__version__",
                             "unknown")
        except Exception:
            out[m] = None
    return out


def reference(prob, Pd, q):
    """A reference optimum that belongs to no solver under test.

    Scoring suboptimality against one solver's own answer makes that solver
    score exactly zero by construction, which is the same defect as scoring a
    one-sided constraint violation against an interior-point method. Here the
    reference is the mean of the values returned by several independent
    high-accuracy solvers, and their spread is reported, so no solver in the
    table is its own yardstick.
    """
    vals = {}
    for s in REF_SOLVERS:
        if s not in qpsolvers.available_solvers:
            continue
        try:
            sol = solve_problem(prob, solver=s, **TOL.get(s, {}))
            if sol.found and sol.x is not None:
                vals[s] = float(0.5 * sol.x @ Pd @ sol.x + q @ sol.x)
        except Exception:
            pass
    if not vals:
        return None, {}, None
    ref = sum(vals.values()) / len(vals)
    spread = max(abs(v - ref) for v in vals.values()) / max(1.0, abs(ref))
    return ref, vals, spread

def run(d, dense, label, out):
    P,q,G,h = (d["P_dense"],d["q"],d["G_dense"],d["h"]) if dense else \
              (d["P"],d["q"],d["G"],d["h"])
    A = d.get("A_dense") if dense else d.get("A")
    b = d["b"] if A is not None else None
    prob = Problem(P,q,G,h,A,b)
    Pd = d["P_dense"] if dense else d["P"].toarray()
    ref, ref_vals, ref_spread = reference(prob, Pd, q)
    print(f"\n=== {label}  (reference f* = {ref:.9f} from "
          f"{len(ref_vals)} independent solvers, spread {ref_spread:.2e}) ===")
    for rs, rv in sorted(ref_vals.items()):
        print(f"      reference {rs:10s} {rv:.12f}")
    print(f"{'solver':10s} {'default ms':>11s} {'tol 1e-9 ms':>12s} {'subopt@def':>12s} {'subopt@tol':>12s}")
    rows={}
    for s in sorted(qpsolvers.available_solvers):
        rec={}
        for mode,kw in (("default",{}),("matched",TOL.get(s,{}))):
            try:
                for _ in range(3): solve_problem(prob, solver=s, **kw)
                ts=[]
                for _ in range(60):
                    t0=time.perf_counter(); sol=solve_problem(prob, solver=s, **kw)
                    ts.append((time.perf_counter()-t0)*1e3)
                if sol.found:
                    f = 0.5*sol.x@Pd@sol.x + q@sol.x
                    rec[mode]=dict(ms=statistics.median(ts),
                                   subopt=abs(f-ref)/max(1.0,abs(ref)))
                else: rec[mode]=dict(ms=statistics.median(ts), subopt=None)
            except Exception as e: rec[mode]=dict(error=type(e).__name__)
        rows[s]=rec
        d0,d1=rec.get("default",{}),rec.get("matched",{})
        f=lambda v: "--" if v is None else (f"{v:.2e}" if isinstance(v,float) else str(v))
        print(f"{s:10s} {d0.get('ms',float('nan')):11.3f} {d1.get('ms',float('nan')):12.3f} "
              f"{f(d0.get('subopt')):>12s} {f(d1.get('subopt')):>12s}")
    out[label]=dict(reference=float(ref),
                    reference_solvers={k: float(v) for k, v in ref_vals.items()},
                    reference_spread=float(ref_spread), eps=EPS,
                    tolerance_settings={k: v for k, v in TOL.items()},
                    rows=rows)

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="matched_tolerance.json")
    a=ap.parse_args(); out={}
    run(problems.srbd_mpc(horizon=40,form="sparse",seed=0), False, "sparse_N40", out)
    run(problems.wbc_id(seed=0), True, "wbc", out)
    out["env"]=dict(platform=platform.platform(),
                    machine=platform.machine(),
                    python=platform.python_version(),
                    qpsolvers=qpsolvers.__version__,
                    versions=solver_versions())
    json.dump(out, open(a.out,"w"), indent=1); print("\nwrote", a.out)

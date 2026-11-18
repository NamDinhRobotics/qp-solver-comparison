"""A whole-body inverse-dynamics QP built from a real robot model.

The synthetic WBC instance of `problems.wbc_id` has the right size,
constraint structure and units, but its mass matrix and contact Jacobian are
random, so its optimum is not a stance: the feet carry 21.7 N against a
117.7 N robot and three of four sit on the lower force bound. This module
builds the same QP from an actual URDF with Pinocchio, so that M, the
nonlinear terms and the contact Jacobians are the robot's own and the optimum
is a real stance.

    python wbc_urdf.py --robot go2 --out wbc_urdf.json

Two things here are easy to get wrong and are therefore checked rather than
assumed.

1. The contact frames. Pinocchio exposes two frames per foot for the Go2 --
   a BODY frame `FL_foot` and a coincident FIXED_JOINT frame `FL_foot_joint`
   -- so a substring filter on "foot" returns eight candidates, and taking
   the first four alphabetically yields `FL_foot, FL_foot_joint, FR_foot,
   FR_foot_joint`: the two front feet, each counted twice, with the hind legs
   never in contact. That duplication makes the stacked Jacobian rank 6 of 12
   and its condition number 1e53, which looks like a robot property and is
   not. We select the four BODY frames by exact suffix and assert that their
   placements are distinct.

2. The posture. The base height does not enter M, does not enter the
   gravity-only nonlinear terms at zero velocity, and does not enter a
   LOCAL_WORLD_ALIGNED frame Jacobian at neutral orientation, so varying it
   produces a byte-identical QP. Postures here are joint configurations, and
   the base is then lowered until the lowest foot touches the ground, so each
   posture is a different stance rather than the same robot at a different
   altitude.

Requires `pin` (Pinocchio) and `robot_descriptions`.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems  # noqa: E402  (for the friction block)

# thigh and calf angles, in radians. The hips stay at zero.
POSTURES = {
    "nominal": (0.80, -1.60),
    "crouched": (1.35, -2.55),
}


def _feet_frames(model):
    """The four foot BODY frames, by exact suffix, with duplicates excluded."""
    import pinocchio as pin
    names = sorted({f.name for f in model.frames
                    if f.name.endswith("_foot")
                    and f.type == pin.FrameType.BODY})
    if len(names) != 4:                      # fall back, then verify
        names = sorted({f.name for f in model.frames
                        if f.name.endswith("_foot")})
    if len(names) != 4:
        raise RuntimeError(f"expected 4 foot frames, found {names}")
    return names


def _set_posture(model, q, thigh, calf):
    """Write one thigh/calf angle pair into every leg of a free-flyer model."""
    import pinocchio as pin
    for jid in range(1, model.njoints):
        name = model.names[jid]
        idx = model.joints[jid].idx_q
        if model.joints[jid].nq != 1:
            continue
        if name.endswith("_thigh_joint"):
            q[idx] = thigh
        elif name.endswith("_calf_joint"):
            q[idx] = calf
        elif name.endswith("_hip_joint"):
            q[idx] = 0.0
    return q


def build(robot="go2", mu=0.6, fz_min=5.0, fz_max=250.0, eps=1e-4,
          configuration="nominal"):
    import pinocchio as pin
    from robot_descriptions.loaders.pinocchio import load_robot_description

    robot_model = load_robot_description(f"{robot}_description",
                                         root_joint=pin.JointModelFreeFlyer())
    model, data = robot_model.model, robot_model.data
    nq, nv = model.nq, model.nv

    thigh, calf = POSTURES[configuration]
    q = pin.neutral(model)
    q = _set_posture(model, q, thigh, calf)

    feet = _feet_frames(model)
    ids = [model.getFrameId(n) for n in feet]

    # put the stance on the ground: lower the base until the lowest foot
    # touches z = 0, so the posture is a stance and not a pose in mid-air
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    foot_z = [float(data.oMf[i].translation[2]) for i in ids]
    q[2] -= min(foot_z)

    v = np.zeros(nv)
    pin.computeAllTerms(model, data, q, v)
    M = np.array(data.M, dtype=float)
    M = 0.5 * (M + M.T)
    nle = np.array(pin.nonLinearEffects(model, data, q, v), dtype=float)

    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    Jc = np.vstack([
        pin.getFrameJacobian(model, data, i, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        for i in ids])
    dJdq = np.zeros(3 * len(ids))

    # the four contacts must be four distinct points
    pts = np.array([data.oMf[i].translation for i in ids])
    if len(np.unique(np.round(pts, 9), axis=0)) != len(ids):
        raise RuntimeError(f"duplicate contact points for frames {feet}")

    n_contacts = len(ids)
    nf = 3 * n_contacts
    n = nv + nf

    rng = np.random.default_rng(0)
    a_des = np.zeros(nv)
    a_des[:6] = rng.normal(0.0, 0.5, 6)
    W = np.diag(np.concatenate([10.0 * np.ones(6), np.ones(nv - 6)]))
    P = np.zeros((n, n))
    P[:nv, :nv] = W
    P[nv:, nv:] = eps * np.eye(nf)
    qv = np.zeros(n)
    qv[:nv] = -W @ a_des

    Aeq = np.zeros((6 + nf, n))
    Aeq[:6, :nv] = M[:6, :]
    Aeq[:6, nv:] = -Jc[:, :6].T
    beq = np.concatenate([-nle[:6], -dJdq])
    Aeq[6:, :nv] = Jc

    Cf, hf = problems._friction_block(mu, fz_min, fz_max)
    n_joints = nv - 6
    Gin = np.zeros((6 * n_contacts + 2 * n_joints, n))
    hin = np.zeros(6 * n_contacts + 2 * n_joints)
    for i in range(n_contacts):
        Gin[6 * i:6 * (i + 1), nv + 3 * i:nv + 3 * (i + 1)] = Cf
        hin[6 * i:6 * (i + 1)] = hf
    tau_max = 33.5
    r0 = 6 * n_contacts
    Tact = np.zeros((n_joints, n))
    Tact[:, :nv] = M[6:, :]
    Tact[:, nv:] = -Jc[:, 6:].T
    Gin[r0:r0 + n_joints, :] = Tact
    hin[r0:r0 + n_joints] = tau_max - nle[6:]
    Gin[r0 + n_joints:, :] = -Tact
    hin[r0 + n_joints:] = tau_max + nle[6:]

    mass = sum(i.mass for i in model.inertias)
    meta = dict(robot=robot, configuration=configuration,
                thigh_rad=thigh, calf_rad=calf,
                base_height_m=float(q[2]), nv=nv, n=n,
                n_eq=Aeq.shape[0], n_in=Gin.shape[0], n_contacts=n_contacts,
                feet=feet,
                contact_points=[[round(float(c), 6) for c in p] for p in pts],
                mass_kg=float(mass), weight_N=float(mass * 9.81),
                cond_M=float(np.linalg.cond(M)),
                cond_Jc=float(np.linalg.cond(Jc)),
                rank_Jc=int(np.linalg.matrix_rank(Jc)),
                rank_Jc_max=3 * n_contacts,
                rank_Aeq=int(np.linalg.matrix_rank(Aeq)),
                rank_Aeq_max=Aeq.shape[0])
    return dict(P=P, q=qv, G=Gin, h=hin, A=Aeq, b=beq, meta=meta)


def diagnose(d, x):
    """Is the reported optimum a stance? Checked, not asserted."""
    m = d["meta"]
    nv, nc = m["nv"], m["n_contacts"]
    f = x[nv:]
    fz = float(sum(f[3 * i + 2] for i in range(nc)))
    vdot = x[:nv]
    return dict(fz_total_N=fz,
                weight_N=m["weight_N"],
                fz_over_weight=fz / m["weight_N"],
                base_lin_acc=[round(float(c), 4) for c in vdot[3:6]],
                base_ang_acc=[round(float(c), 4) for c in vdot[0:3]],
                max_eq_residual=float(np.max(np.abs(d["A"] @ x - d["b"]))),
                max_ineq_violation=float(np.max(d["G"] @ x - d["h"])))


def environment():
    env = dict(platform=platform.platform(), machine=platform.machine(),
               python=platform.python_version(), numpy=np.__version__)
    for mod in ("pinocchio", "qpsolvers", "robot_descriptions", "osqp",
                "proxsuite", "piqp", "daqp", "scs", "clarabel", "qpalm",
                "ecos", "cvxopt", "quadprog", "highspy"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            env[mod] = None
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="go2")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", default="wbc_urdf.json")
    ap.add_argument("--export", default=None,
                    help="also write the raw QP data for another environment")
    a = ap.parse_args()

    import qpsolvers
    from qpsolvers import Problem, solve_problem

    def time_rows(prob, d=None, repeats=a.repeats):
        rows = {}
        for s in sorted(qpsolvers.available_solvers):
            try:
                for _ in range(3):
                    solve_problem(prob, solver=s)
                ts = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    sol = solve_problem(prob, solver=s)
                    ts.append((time.perf_counter() - t0) * 1e3)
                ts.sort()
                ok = bool(getattr(sol, "found", False))
                rec = dict(median_ms=statistics.median(ts),
                           p99_ms=ts[int(0.99 * (len(ts) - 1))], found=ok)
                if ok and sol.x is not None and d is not None:
                    rec.update(diagnose(d, sol.x))
                    rec["objective"] = float(
                        0.5 * sol.x @ d["P"] @ sol.x + d["q"] @ sol.x)
                rows[s] = rec
            except Exception as exc:
                rows[s] = dict(error=f"{type(exc).__name__}: {exc}")
        return rows

    out = {"env": environment(), "instances": {}}

    # the synthetic instance of problems.wbc_id, measured on this machine with
    # this solver set, so that the ranking comparison of the paper is not made
    # across two platforms
    sd = problems.wbc_id(seed=0)
    sp_ = Problem(sd["P"].toarray() if hasattr(sd["P"], "toarray") else sd["P"],
                  sd["q"],
                  sd["G"].toarray() if hasattr(sd["G"], "toarray") else sd["G"],
                  sd["h"],
                  sd["A"].toarray() if hasattr(sd["A"], "toarray") else sd["A"],
                  sd["b"])
    print("=== synthetic problems.wbc_id (same machine, same solver set) ===")
    srows = time_rows(sp_)
    for s, r in sorted(srows.items(),
                       key=lambda kv: kv[1].get("median_ms", 9e9)):
        if "error" in r:
            print(f"    {s:10s} ERROR {r['error'][:52]}")
        else:
            print(f"    {s:10s} {r['median_ms']:8.4f} ms "
                  f"(p99 {r['p99_ms']:8.4f})  found={r['found']}")
    out["synthetic"] = dict(meta=sd.get("meta", {}), rows=srows)

    export = {}
    for cfg in POSTURES:
        d = build(robot=a.robot, configuration=cfg)
        m = d["meta"]
        print(f"\n=== {a.robot} / {cfg} ===")
        print(f"  feet={m['feet']}")
        print(f"  n={m['n']} n_eq={m['n_eq']} n_in={m['n_in']}  "
              f"mass={m['mass_kg']:.2f} kg (weight {m['weight_N']:.1f} N)  "
              f"base z={m['base_height_m']:.3f} m")
        print(f"  cond(M)={m['cond_M']:.4g}  cond(Jc)={m['cond_Jc']:.4g}  "
              f"rank(Jc)={m['rank_Jc']}/{m['rank_Jc_max']}  "
              f"rank(Aeq)={m['rank_Aeq']}/{m['rank_Aeq_max']}")

        prob = Problem(d["P"], d["q"], d["G"], d["h"], d["A"], d["b"])
        rows = time_rows(prob, d)
        for s, r in sorted(rows.items(),
                           key=lambda kv: kv[1].get("median_ms", 9e9)):
            if "error" in r:
                print(f"    {s:10s} ERROR {r['error'][:52]}")
            else:
                print(f"    {s:10s} {r['median_ms']:8.4f} ms "
                      f"(p99 {r['p99_ms']:8.4f})  found={r['found']}  "
                      f"f_z={r.get('fz_total_N', float('nan')):7.2f} N "
                      f"({r.get('fz_over_weight', float('nan')):.4f} W)")
        out["instances"][cfg] = dict(meta=m, rows=rows)
        if a.export:
            export[cfg] = {k: np.asarray(d[k]).tolist()
                           for k in ("P", "q", "G", "h", "A", "b")}

    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)
    if a.export:
        json.dump(export, open(a.export, "w"))
        print("wrote", a.export)


if __name__ == "__main__":
    main()

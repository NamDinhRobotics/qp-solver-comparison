"""Re-check the ProxQP default-threshold finding against any installed proxsuite.

    python proxqp_version_check.py out.json

Solves the condensed MPC instances at *genuinely untouched* defaults --- only
`verbose` is set --- and then repeats them with `eps_primal_inf` lowered and
nothing else changed. For each horizon it prints the returned status, the
iteration count, the worst constraint violation of the returned point and the
objective, so a reader can confirm or refute the finding on their own version
and platform in one command.
"""

import os
import sys
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems as pb  # noqa: E402
import proxsuite  # noqa: E402
import importlib.metadata as md  # noqa: E402

ver = md.version("proxsuite")
print(f"=== proxsuite {ver} on {sys.platform} ===\n")
out = {"version": ver, "platform": sys.platform, "rows": []}

for N in (10, 20, 30, 40):
    d = pb.srbd_mpc(horizon=N, form='condensed', seed=0)
    P, q, G, h = d['P_dense'], d['q'], d['G_dense'], d['h']
    n, ni = P.shape[0], G.shape[0]
    # (a) TRUE defaults: touch nothing at all
    qp = proxsuite.proxqp.dense.QP(n, 0, ni)
    qp.init(P, q, None, None, G, None, h)
    qp.settings.verbose = False
    qp.solve()
    st = str(qp.results.info.status).split('.')[-1]
    x = qp.results.x
    viol = float(np.max(G @ x - h))
    obj = float(0.5 * x @ P @ x + q @ x)
    print(f"N={N:2d} TRUE DEFAULTS      -> {st:32s} iter={qp.results.info.iter:5d} "
          f"maxviol={viol:+9.3f} obj={obj:.6f}")
    out["rows"].append(dict(N=N, mode="true_defaults", status=st,
                            iter=int(qp.results.info.iter), viol=viol, obj=obj))
    # (b) only eps_primal_inf lowered, everything else default
    qp2 = proxsuite.proxqp.dense.QP(n, 0, ni)
    qp2.init(P, q, None, None, G, None, h)
    qp2.settings.verbose = False
    qp2.settings.eps_primal_inf = 1e-6
    qp2.solve()
    st2 = str(qp2.results.info.status).split('.')[-1]
    x2 = qp2.results.x
    print(f"      eps_primal_inf=1e-6 -> {st2:32s} iter={qp2.results.info.iter:5d} "
          f"maxviol={np.max(G@x2-h):+9.3e} obj={0.5*x2@P@x2+q@x2:.6f}")
    out["rows"].append(dict(N=N, mode="eps_pinf_1e-6", status=st2,
                            iter=int(qp2.results.info.iter),
                            viol=float(np.max(G@x2-h)),
                            obj=float(0.5*x2@P@x2+q@x2)))
json.dump(out, open(sys.argv[1], 'w'), indent=1)

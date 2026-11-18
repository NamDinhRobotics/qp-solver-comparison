"""Export the benchmark instances to plain-text files for the C++ harness.

The C++ solvers (qpOASES, qpmad, eiquadprog) are dense methods, so we export
the dense problems: the condensed MPC instances and the WBC instance. Format:

    n n_eq n_in
    P   (n*n, row-major)
    q   (n)
    A   (n_eq*n, row-major; omitted when n_eq == 0)
    b   (n_eq)
    G   (n_in*n, row-major)
    h   (n_in)

All in ASCII with 17 significant digits, which round-trips an IEEE double
exactly, so the C++ side solves bit-identical problems to the Python side.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems  # noqa: E402


def dump(path, d):
    P = d["P_dense"]
    G = d["G_dense"]
    A = d.get("A_dense")
    q = d["q"]
    h = d["h"]
    b = d["b"] if A is not None else np.zeros(0)
    n = P.shape[0]
    n_eq = 0 if A is None else A.shape[0]
    n_in = G.shape[0]
    with open(path, "w") as f:
        f.write(f"{n} {n_eq} {n_in}\n")
        for M in (P,):
            np.savetxt(f, M.reshape(1, -1), fmt="%.17g")
        np.savetxt(f, q.reshape(1, -1), fmt="%.17g")
        if n_eq:
            np.savetxt(f, A.reshape(1, -1), fmt="%.17g")
            np.savetxt(f, b.reshape(1, -1), fmt="%.17g")
        np.savetxt(f, G.reshape(1, -1), fmt="%.17g")
        np.savetxt(f, h.reshape(1, -1), fmt="%.17g")
    return dict(name=os.path.basename(path), n=n, n_eq=n_eq, n_in=n_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="problems_txt")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    index = []
    for N in (10, 20, 30, 40):
        d = problems.srbd_mpc(horizon=N, form="condensed", seed=0)
        index.append(dump(os.path.join(a.outdir,
                                       f"srbd_mpc_condensed_N{N}.txt"), d))
    d = problems.wbc_id(seed=0)
    index.append(dump(os.path.join(a.outdir, "wbc_id.txt"), d))

    for row in index:
        print(row)


if __name__ == "__main__":
    main()

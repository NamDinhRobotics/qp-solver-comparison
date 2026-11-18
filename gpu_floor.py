"""Measure the fixed cost of using a GPU at all, for problems of our size.

A WBC QP here solves in about 0.023 ms on one CPU core. Before asking whether
a GPU solver would be faster, it is worth knowing what a GPU round trip costs
before any arithmetic happens. We measure three floors:

  * kernel launch + synchronize, on a trivially small array;
  * host -> device -> host transfer of the WBC problem data;
  * a dense 30x30 matrix factorization on the GPU, the smallest useful
    linear-algebra step inside a QP iteration.

None of these is a QP solve. They are measured through CuPy, so they include
Python dispatch and a fresh device allocation per call: a compiled CUDA solver
with pinned memory and a persistent stream would pay less. They are therefore
an UPPER bound on the device's fixed cost, not a lower one. That cuts against
the conclusion drawn from them, not for it: a native implementation would pay
less than this, so the margin between the round trip and the CPU solve would
shrink. The CPU Cholesky baseline is likewise a numpy call, a large share of
which is dispatch rather than arithmetic, which inflates the batch crossover
in the GPU's favour. Both caveats are stated in the paper.
"""

import json
import statistics
import sys
import time

import numpy as np

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    print(json.dumps({"status": "cupy_unavailable", "error": str(exc)}))
    sys.exit(0)


def timeit(fn, reps=2000, warmup=200):
    for _ in range(warmup):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(median_ms=statistics.median(ts),
                p99_ms=ts[min(len(ts) - 1, int(0.99 * (len(ts) - 1)))],
                min_ms=ts[0])


def main():
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    out = {"status": "ok",
           "gpu": props["name"].decode(),
           "cupy": cp.__version__,
           "runtime_version": cp.cuda.runtime.runtimeGetVersion()}

    # --- 1. bare kernel launch + sync -----------------------------------
    tiny = cp.ones(8, dtype=cp.float64)
    out["kernel_launch"] = timeit(lambda: tiny * 2.0)

    # --- 2. round trip of the WBC problem data --------------------------
    # n = 30, n_eq = 18, n_in = 48  ->  P, A, G, and the vectors
    nbytes = (30 * 30 + 18 * 30 + 48 * 30 + 30 + 18 + 48) * 8
    host = np.ones(nbytes // 8, dtype=np.float64)

    def roundtrip():
        d = cp.asarray(host)
        _ = cp.asnumpy(d)

    out["roundtrip_wbc_data"] = timeit(roundtrip, reps=1000, warmup=100)
    out["roundtrip_bytes"] = nbytes

    # --- 3. the smallest useful linear algebra inside a QP iteration ----
    A = np.eye(30) * 3.0 + np.random.default_rng(0).normal(0, 0.01, (30, 30))
    A = A @ A.T
    dA = cp.asarray(A)
    out["cholesky_30x30_gpu"] = timeit(lambda: cp.linalg.cholesky(dA),
                                       reps=1000, warmup=100)

    # the same factorization on the CPU, for scale
    ts = []
    for _ in range(2000):
        t0 = time.perf_counter()
        np.linalg.cholesky(A)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    out["cholesky_30x30_cpu"] = dict(median_ms=statistics.median(ts),
                                     min_ms=ts[0])

    # --- 4. where the GPU does start to pay: batched factorizations -----
    batch = {}
    for B in (1, 8, 64, 512, 4096):
        stack = cp.asarray(np.repeat(A[None, :, :], B, axis=0))
        r = timeit(lambda: cp.linalg.cholesky(stack), reps=200, warmup=50)
        cpu_each = out["cholesky_30x30_cpu"]["median_ms"]
        batch[str(B)] = dict(total_ms=r["median_ms"],
                             per_problem_ms=r["median_ms"] / B,
                             cpu_serial_ms=cpu_each * B,
                             speedup_vs_cpu_serial=(cpu_each * B) / r["median_ms"])
        del stack
        cp.get_default_memory_pool().free_all_blocks()
    out["batched_cholesky"] = batch

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

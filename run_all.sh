#!/usr/bin/env bash
# Regenerate every number, table and figure in Sect. 5 of the paper.
#
#   bash run_all.sh [workdir]
#
# Stage 1 (Python) needs: numpy scipy matplotlib qpsolvers and the solver
# packages osqp proxsuite quadprog scs clarabel daqp piqp qpalm highspy ecos
# cvxopt.  Stage 2 (C++) additionally needs cmake, a C++17 compiler and
# network access to clone Eigen, qpmad, qpOASES, qpSWIFT and eiquadprog.
#
# Neither stage needs privileged access. Energy counters are NOT read: on a
# stock Linux the RAPL sysfs files are root-only, which is why the paper
# reports no Solve Frequency per Watt of its own.
set -e
WORK="${1:-$PWD/qpbench_run}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"          # absolute, so a relative argument works
cd "$WORK"

# The measurement conditions reported in the paper. Without these the timings
# are not the ones we report: BLAS threading changes the standing of the
# dense/LAPACK-heavy solvers, and core pinning decides whether a hybrid CPU
# runs the loop on a performance or an efficiency core.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PIN=""
if command -v taskset >/dev/null 2>&1; then
    PIN="taskset -c ${QPBENCH_CPUS:-0-5}"   # performance cores; override as needed
fi

echo "### stage 1: Python benchmark (11 solvers)"
$PIN "$PY" "$HERE/run_bench.py" --repeats 100 --out "$WORK/results.json"

echo "### stage 2: native warm start through each solver's own API"

echo "### stage 3: export instances for the C++ harness"
"$PY" "$HERE/export_problems.py" --outdir "$WORK/problems_txt"

echo "### stage 4: build and run the C++ harness (qpmad, eiquadprog, qpOASES)"
# stage 4 is allowed to fail without losing stages 5 and 6
set +e
CPP_ROOT="$WORK/cpp" bash "$HERE/build_deps.sh" && \
CPP_ROOT="$WORK/cpp" bash "$HERE/build_cpp_bench.sh" && \
$PIN "$WORK/cpp/cpp_bench" 100 "$WORK"/problems_txt/*.txt \
    > "$WORK/cpp_results.json"
[ $? -eq 0 ] || echo "   stage 4 failed; continuing with the Python results only"
set -e

echo "### stage 4b: re-check the ProxQP finding on the installed proxsuite"
"$PY" "$HERE/proxqp_version_check.py" "$WORK/proxqp_recheck.json" || \
    echo "   proxsuite not installed; skipping"

echo "### stage 4c: controlled warm-start, matched tolerance, URDF WBC"
$PIN "$PY" "$HERE/warmstart_controlled.py" --out "$WORK/warmstart_controlled.json" || true
$PIN "$PY" "$HERE/matched_tolerance.py"    --out "$WORK/matched_tolerance.json"    || true
"$PY" "$HERE/wbc_urdf.py" --robot go2      --out "$WORK/wbc_urdf.json"             || \
    echo "   pinocchio/robot_descriptions not installed; skipping the URDF check"

echo "### stage 4d: seed study (how much of the ordering survives a redraw)"
$PIN "$PY" "$HERE/seed_study.py" --seeds 0,1,2,3,4 --out "$WORK/seed_study.json" || true

echo "### stage 5: GPU floor (optional; needs CuPy, skipped if absent)"
if "$PY" "$HERE/gpu_floor.py" > "$WORK/gpu_floor.json" && \
   ! grep -q cupy_unavailable "$WORK/gpu_floor.json"; then
    echo "   GPU floor measured"
else
    echo "   no CuPy / no GPU: the GPU table cannot be regenerated"
    rm -f "$WORK/gpu_floor.json"
fi

echo "### stage 6: tables and figures"
"$PY" "$HERE/make_tables.py" --results "$WORK/results.json" --wsctl "$WORK/warmstart_controlled.json" --matched "$WORK/matched_tolerance.json" --outdir "$WORK/out"
"$PY" - <<EOF
import sys; sys.path.insert(0, "$HERE")
import make_tables
make_tables.emit_native_table("$WORK/results.json", "$WORK/cpp_results.json",
                              "$WORK/out")
import os
g = "$WORK/gpu_floor.json"
if os.path.exists(g):
    make_tables.emit_gpu_table(g, "$WORK/cpp_results.json", "$WORK/out")
EOF
"$PY" "$HERE/make_figures.py" --results "$WORK/results.json" \
    --cpp "$WORK/cpp_results.json" \
    --wsctl "$WORK/warmstart_controlled.json" \
    --outdir "$WORK/out"

echo "### done. LaTeX fragments and the figure are in $WORK/out"

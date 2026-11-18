#!/usr/bin/env bash
# Compile the native C++ timing harness. Release flags are essential: an
# unoptimised build of these solvers is slower by more than an order of
# magnitude and the timings would be meaningless.
set -ex
ROOT="${CPP_ROOT:-$PWD/cpp}"
SRC="$ROOT/src"
HERE="$(cd "$(dirname "$0")" && pwd)"
g++ -O3 -DNDEBUG -march=native -std=c++17 \
    "$HERE/cpp_bench.cpp" \
    "$SRC/eiquadprog/src/eiquadprog-fast.cpp" \
    -I "$SRC/eigen" \
    -I "$SRC/qpmad/include" \
    -I "$SRC/eiquadprog/include" \
    -I "$ROOT/install/include" \
    -L "$ROOT/install/lib" \
    -lqpOASES \
    -o "$ROOT/cpp_bench"
echo "BUILD_OK"
"$ROOT/cpp_bench" 2>&1 | head -3 || true

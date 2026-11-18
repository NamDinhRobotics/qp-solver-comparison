#!/usr/bin/env bash
# Fetch and build the C/C++ QP solvers that have no ready Python binding,
# entirely inside $CPP_ROOT (default ./cpp). Nothing is installed system-wide
# and no privileged access is used.
set -x
ROOT="${CPP_ROOT:-$PWD/cpp}"
PREFIX="$ROOT/install"
mkdir -p "$ROOT/src" "$PREFIX"
cd "$ROOT/src" || exit 1

# Pinned so that a later reader measures the same code we did. Update
# deliberately, not by accident.
clone () {
  if [ ! -d "$2" ]; then
    git clone "$1" "$2"
    if [ -n "$3" ]; then ( cd "$2" && git checkout --quiet "$3" ); fi
    ( cd "$2" && echo "$2 $(git rev-parse HEAD)" >> "$ROOT/versions.txt" )
  fi
}
: > "$ROOT/versions.txt" 2>/dev/null || true

clone https://gitlab.com/libeigen/eigen.git eigen "${EIGEN_REF:-}"
clone https://github.com/asherikov/qpmad.git qpmad "${QPMAD_REF:-}"
clone https://github.com/coin-or/qpOASES.git qpOASES "${QPOASES_REF:-}"
clone https://github.com/stack-of-tasks/eiquadprog.git eiquadprog "${EIQUADPROG_REF:-}"

ls "$ROOT/src/eigen/Eigen/Core" && echo "EIGEN_OK"
ls "$ROOT/src/qpmad/include/qpmad/solver.h" && echo "QPMAD_OK"

# ---- qpOASES: static library ----
cd "$ROOT/src/qpOASES" || exit 1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_SHARED_LIBS=OFF
cmake --build build -j8 && cmake --install build || echo "FAILED:qpOASES"


echo "=== installed artifacts ==="
find "$PREFIX" -maxdepth 4 \( -name '*.a' -o -name '*.so*' -o -name '*.h' -o -name '*.hpp' \) | head -60

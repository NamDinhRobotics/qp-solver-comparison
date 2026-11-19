# Convex QP solvers for real-time control — reproduction package

Software and data behind *Formulation, defaults, and interfaces: a
reproducible computational comparison of convex QP solvers for real-time
control*.

Fourteen convex quadratic-programming solvers are measured on two problem
templates from a legged-robot control stack — single-rigid-body
model-predictive control, in sparse optimal-control and condensed dense form,
and whole-body inverse dynamics — for solve time, attained constraint
residual, suboptimality at a matched tolerance, warm-start benefit and
infeasibility behaviour.

Every number, table and figure in the manuscript is produced by the code here
from the raw result files here. MIT licensed.

- Preprint of an earlier version: [arXiv:2510.21773](https://arxiv.org/abs/2510.21773)
- Manuscript status: under review

## Verify the published tables without re-measuring

The raw results of the reported run are committed, so all nine table
fragments in the manuscript can be regenerated and checked without running a
single solver. This needs only Python, NumPy and Matplotlib:

```
pip install numpy matplotlib
python3 make_tables.py --results results.json \
        --wsctl warmstart_controlled.json \
        --matched matched_tolerance.json --outdir out
```

`tab_native.tex` and `tab_gpu.tex` come from two further entry points in the
same module:

```
python3 -c "import make_tables as m; \
  m.emit_native_table('results.json','cpp_results.json','out'); \
  m.emit_gpu_table('gpu_floor.json','cpp_results.json','out')"
```

The nine fragments reproduce byte for byte against the manuscript sources.

## Re-measure from scratch

```
bash run_all.sh [workdir]        # default workdir: ./qpbench_run
```

`run_all.sh` runs the whole pipeline and writes the LaTeX table fragments and
figures to `<workdir>/out`. It exports the measurement conditions used for the
reported timings (`OMP_NUM_THREADS=1`, a single BLAS thread, and core pinning
via `taskset` where available): without these the timings are not the ones
reported, because BLAS threading changes the standing of the dense solvers and
core pinning decides whether the loop runs on a performance or an efficiency
core.

No stage needs privileged access. No energy figure is reported for the same reason. On a stock Linux the Intel RAPL counters under
`/sys/class/powercap` are root-only.

| stage | script | needs |
|---|---|---|
| 1 | `run_bench.py` | Python, `qpsolvers` + the solver packages |
| 3 | `export_problems.py` | Python |
| 4 | `build_deps.sh`, `build_cpp_bench.sh`, `cpp_bench.cpp` | CMake, C++17, network access to clone Eigen / qpmad / qpOASES / qpSWIFT / eiquadprog |
| 4b | `proxqp_version_check.py` | `proxsuite` |
| 4c | `warmstart_controlled.py`, `matched_tolerance.py`, `wbc_urdf.py` | Python; `pinocchio` + `robot_descriptions` for the URDF check |
| 4d | `seed_study.py` | Python |
| 5 | `gpu_floor.py` | CuPy and a CUDA device (skipped if absent) |
| 6 | `make_tables.py`, `make_figures.py` | Python, Matplotlib |

Stages 4 and 5 may fail without losing the rest. Install the Python side with
`pip install -r requirements.txt`.

## What each file is

**Generators and harnesses**

| file | what it does |
|---|---|
| `problems.py` | the instance generator: both QP templates, seeded, so no data files are needed |
| `run_bench.py` | stage-1 harness — every solver reachable through `qpsolvers`, at default settings |
| `cpp_bench.cpp` | native harness for `qpmad`, `eiquadprog`, `qpOASES`, reading the same instances exported by `export_problems.py` |
| `warmstart_controlled.py` | the warm-start decomposition: cold / reset / warm along a closed-loop receding-horizon sequence |
| `matched_tolerance.py` | suboptimality at a matched tolerance, against a reference computed by solvers not under test in that comparison |
| `wbc_urdf.py` | rebuilds the whole-body QP from a published quadruped URDF via Pinocchio, in two postures |
| `seed_study.py` | repeats three families over five seeds and reports rank intervals |
| `gpu_floor.py` | device round-trip floor; reports `cupy_unavailable` rather than failing |
| `proxqp_version_check.py` | one-command re-check of the false infeasibility-certificate finding against the installed `proxsuite` |

The reset leg in `warmstart_controlled.py` is **per solver**, because the APIs
differ in what an update retains: `daqp` keeps its working set unless a zeroed
`sense` array is passed, and `piqp` has no warm-start setting at all, so its
warm leg does not exist and is reported as absent rather than as `1.00x`.
See the module docstring.

**Raw results** — inputs to the tables and figures

| file | contents |
|---|---|
| `results.json` | stage-1 Python timings and solution quality, plus the warm-start, infeasibility, overhead and ProxQP-threshold probes; records its own environment block |
| `cpp_results.json` | native harness timings. **Known gap:** does not record its host, and the manuscript states this |
| `warmstart_controlled.json` | warm-start decomposition over five perturbation sizes |
| `matched_tolerance.json` | matched-tolerance suboptimality |
| `wbc_urdf.json` | the URDF-derived whole-body comparison |
| `seed_study.json` | the five-seed rank intervals |
| `gpu_floor.json` | device round-trip floor |
| `proxqp_recheck_macos.json` | the version re-check |

The four files added most recently each record their own environment block.

**Emitters** — `make_tables.py` (nine table fragments) and `make_figures.py`
(five figures). Both read only the JSON files, never the solvers.

## Known reproducibility gaps

Both are stated in the manuscript as well, and in `requirements.txt`:

1. The C++ dependencies are cloned from upstream and, in the run reported
   here, were not pinned to a commit. The scripts now record and accept refs.
2. Three Python solver versions were not captured by that run.

## Citing

See `CITATION.cff`. Until the paper appears, please cite the preprint
[arXiv:2510.21773](https://arxiv.org/abs/2510.21773).

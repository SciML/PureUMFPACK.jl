# PureUMFPACK.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.sciml.ai/PureUMFPACK/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.sciml.ai/PureUMFPACK/dev/)
[![CI](https://github.com/SciML/PureUMFPACK.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SciML/PureUMFPACK.jl/actions/workflows/CI.yml)
[![Build Status](https://github.com/SciML/PureUMFPACK.jl/actions/workflows/Downgrade.yml/badge.svg)](https://github.com/SciML/PureUMFPACK.jl/actions/workflows/Downgrade.yml)
[![codecov](https://codecov.io/gh/SciML/PureUMFPACK.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/PureUMFPACK.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A **pure-Julia** unsymmetric sparse LU factorization, in the spirit of SuiteSparse
UMFPACK. No `ccall`, no C library — ordering, scaling, symbolic + numeric
factorization, and triangular solves are all implemented in Julia. The result
follows UMFPACK's exact convention:

```
(F.Rs .* A)[F.p, F.q] == F.L * F.U
```

with `L` unit lower triangular and `U` upper triangular, so it is a drop-in
analogue of `SparseArrays.lu` for validation and benchmarking.

## Pieces

| Piece | File | Description |
|-------|------|-------------|
| Ordering | `src/amd.jl` | Pure-Julia **AMD** (approximate minimum degree), a faithful port of CSparse `cs_amd`. `amd_order_sym` (on A+Aᵀ) and `colamd_order` (AMD on the AᵀA column-intersection graph = `cs_amd(order=2)`). |
| Row scaling | `src/scaling.jl` | `SCALE_SUM` (UMFPACK default), `SCALE_MAX`, `SCALE_NONE`. |
| Numeric (default) | `src/gplu.jl` | **Gilbert–Peierls** left-looking LU with threshold partial pivoting; depth-first reachability for each column's structure; preallocated, geometrically-grown factor storage; allocation-free column canonicalization. |
| Symbolic analysis | `src/symbolic.jl` | Elimination tree, postorder, column structure of L, fundamental-supernode amalgamation — feeds the multifrontal kernel. |
| Numeric (multifrontal) | `src/multifrontal.jl` | **Supernodal multifrontal** LU (UMFPACK's algorithm): dense frontal matrices, child contribution blocks assembled by extend-add, pivot blocks factored via LAPACK `getrf` and updated with `trsm`/`gemm` — **BLAS-3** dense fronts. In-block partial pivoting. |
| Triangular solves | `src/solve.jl` | Column-oriented forward/backward CSC solves. |
| High-level API | `src/interface.jl` | `splu`, `solve`/`\`, iterative refinement. |

## Usage

```julia
using PureUMFPACK, SparseArrays, LinearAlgebra

A = sprand(10_000, 10_000, 5/10_000) + 10I
b = randn(10_000)

F = splu(A)                      # ordering=:amd, scale=SCALE_SUM, tol=0.1
x = F \ b                        # or solve(F, b)
x = solve(F, b; refine = 2)      # 2 steps of iterative refinement

F = splu(A; ordering = :colamd, tol = 1.0, scale = SCALE_MAX)

# supernodal multifrontal (BLAS-3) — much faster on 3D / dense-front problems;
# best for SPD / diagonally-dominant / structurally-symmetric systems
F = splu(A; method = :multifrontal)
x = F \ b

# low-level pieces
q = amd_order_sym(A)             # symmetric AMD permutation
q = colamd_order(A)              # column ordering (AᵀA)
G = gplu(A; q = q, tol = 0.1)    # raw L,U,p,q with  A[p,q] == L*U
M = multifrontal_lu(A)           # raw multifrontal factorization
```

## Validation

`amd_order_sym` reproduces the **exact** fill of the SuiteSparse `AMD` package
(ratio 1.000 across Poisson 2D/3D, arrowhead and random matrices; marginally lower
on random), and `gplu` reproduces UMFPACK's fill exactly given the same column
ordering. The full suite (`test/runtests.jl`, **206 tests, all passing**) checks
reconstruction `A[p,q]=LU`, strict partial pivoting, factor canonicalization
(sorted == double-transpose), every ordering × scaling combination, agreement with
UMFPACK's solution, iterative refinement, edge cases, type generality (ComplexF64,
Float32, Int32 indices, singularity detection), and the multifrontal kernel
(symbolic-fill identity, reconstruction, UMFPACK agreement). See
`results/SUMMARY.md` for the full verified numbers.

## Benchmarks (AMD EPYC 7502, Julia 1.12, single-threaded BLAS)

From `bench/bench_full.jl` → `results/bench_full.log`. Ratios are to UMFPACK:
`numfx` = numeric-only factor time ÷ UMFPACK's full factor time; `totalx` includes
ordering; `solvex` = triangular-solve time ÷ UMFPACK's solve. `fill` matches
UMFPACK exactly on the structured cases (verified) and is **lower** on random.

| matrix | n | fill | numfx | totalx | solvex |
|---|---|---|---|---|---|
| poisson2d-50 | 2500 | = UMF | 0.96 | 1.09 | **0.60** |
| poisson2d-100 | 10000 | = UMF | 1.50 | 1.54 | **0.67** |
| poisson2d-150 | 22500 | = UMF | 1.75 | 1.88 | **0.92** |
| poisson3d-16 | 4096 | = UMF | 4.64 | 4.65 | **0.82** |
| poisson3d-24 | 13824 | = UMF | 9.27 | 9.12 | **0.92** |
| rand-5k-d10 | 5000 | **0.80×** | 13.2 | 13.4 | **0.68** |
| arrowband-5k | 5000 | = UMF | **0.12** | **0.26** | **0.56** |

(The box is shared with other workloads, so factor-time ratios carry ±20% run-to-run
noise; fill and residuals are exact and reproducible.)

### Multifrontal reaches parity with C UMFPACK on 3D (`bench/bench_mf_clean.jl`)

Factorization time as a ratio to UMFPACK on 3D Poisson — the case where the unblocked
GP-LU kernel was 4–20× behind. `mf` = `splu(method=:multifrontal)`. Fill is identical
to UMFPACK and the solve residual is ~1e-15 throughout. `mf×UMF` from the
noise-resistant harness `bench/bench_mf_clean.jl` (interleaved, min-of-8, 1 BLAS
thread, `results/bench_mf_clean.log`); the COO and direct-CSC columns are from prior
logs, shown to trace the two optimization passes.

| matrix | n | original COO ×UMF | + direct-CSC ×UMF | + O(1) alloc (now) ×UMF |
|---|---|---|---|---|
| poisson3d-16 | 4096 | 1.59 | 0.85 | **0.70** |
| poisson3d-24 | 13824 | 1.39 | 0.90 | **0.72** |
| poisson3d-30 | 27000 | 1.79 | 1.23 | **0.94** |
| poisson3d-36 | 46656 | 2.21 | 1.20 | **0.99** |

Two measured optimization passes got here. **(1) Direct-CSC factor construction** —
build L and U straight into preallocated CSC (column pointers sized from the symbolic
structure, numerics scattered to known offsets) instead of the old COO `push!` →
`sparse()` → `[prow,:]` → `_sortcols` path — removed the ~66–78 % factor-bookkeeping
cost (`results/PERF_DIAGNOSIS.md`). **(2) O(1) allocation** — one reused dense front
workspace plus a single LIFO arena for child contribution blocks, instead of a
`zeros(nf,nf)` and a `Matrix` copy per supernode — cut the GC tax that had made the
ratio *degrade with size* (`results/SCALING_DIAGNOSIS.md`): allocation dropped ~4× (3.26
GiB → 787 MiB at n=46656) and GC roughly halved (sustained-loop 13–35 % → 10–29 %; ~3–13 %
single-run). Net effect (clean bench): the kernel now **runs faster than C UMFPACK at
every 3D size tested (0.70–0.99×)**, well below the prior 0.85–1.20× that climbed past
parity. The ratio still drifts mildly upward with n (0.71 → 0.98 across n=1.7k–47k in
`results/mf_scaling.log`) — mf's numeric growth slope remains slightly above UMFPACK's,
so the remaining ~8–13 % large-n GC is the next lever — but it now stays under 1.0
throughout the tested range instead of crossing it. The dense BLAS-3 phase was already
at/above the dense-`lu` ceiling and measured to need no change. GP-LU stays the default
for general unsymmetric stability (multifrontal uses in-block pivoting).

**Where each piece stands**

- **Ordering** — matches SuiteSparse AMD fill exactly; negligible time (≤14 ms even
  at n=22500). Optimized.
- **Triangular solve** — consistently **faster than UMFPACK** (0.56–0.92×). Optimized.
- **Numeric (2D / banded)** — parity with UMFPACK on 2D (≈1–1.8×), and *faster* on
  banded (arrowband 0.12× numeric, 0.26× total). The work is BLAS-1-class scalar
  scatter-axpy with no dense structure to exploit. Optimized for this algorithm.
- **Memory** — preallocated, geometrically-grown factor storage replaced `push!`
  (poisson2d-100 natural order: 108 → 94 MiB, same fill); the new in-place
  merge/insertion column sort dropped sort allocation overhead to **zero** (AMD
  poisson2d-100: 12.9 MiB with sort == 12.9 MiB without). Optimized.
- **Numeric (3D / dense-front)** — the **supernodal multifrontal** kernel
  (`method=:multifrontal`) routes dense frontal updates through BLAS-3 and builds the
  factor directly into preallocated CSC, making it ~5–20× faster than GP-LU on 3D
  Poisson and bringing the time to **0.70–0.99× of the C UMFPACK** (faster than the C
  library at every 3D size tested; ratio drifts mildly with n but stays <1.0). Same
  fill, machine-precision residual. Diagnoses + fixes in `results/PERF_DIAGNOSIS.md` and
  `results/SCALING_DIAGNOSIS.md`. GP-LU remains the default for general unsymmetric
  stability.

## Algorithm notes

- **Gilbert–Peierls** (1988): column `k` is computed by solving `L x = A(:,q[k])`;
  the nonzero pattern of `x` is the set of nodes reachable from the nonzeros of
  `A(:,q[k])` in the DAG of `L`, found by a depth-first search in topological order;
  the numeric solve runs over exactly that pattern. Threshold partial pivoting
  picks the largest un-pivoted entry, preferring the diagonal column within `tol`.
- **AMD** is the quotient-graph approximate minimum degree of Amestoy, Davis & Duff,
  ported from CSparse with element absorption, mass elimination, aggressive
  absorption, supervariable (hash) detection, and assembly-tree postordering.
- **COLAMD** here is AMD applied to the pattern of `AᵀA` with dense rows of `A`
  dropped — the column fill-reducing ordering used by `cs_amd(order=2)`.
- **Multifrontal** (`src/multifrontal.jl`): supernodal LU over the elimination
  tree. Each supernode owns a dense frontal matrix; children's Schur (contribution)
  blocks assemble into the parent by extend-add; the pivot block is factored with
  LAPACK `getrf` and the off-diagonal blocks updated with `trsm`/`gemm`, so the
  dense-front flops run through BLAS-3. Pivoting is restricted to within each
  supernode (the L factor is accumulated in global-row coordinates, then permuted
  to factor order), which is stable for SPD / diagonally-dominant / structurally
  symmetric systems. It does **not** do cross-front threshold (delayed) pivoting.

## Status / next steps

Correct, tested (206/206), and benchmarked piece-by-piece. The pure-Julia pipeline
matches SuiteSparse AMD fill exactly, has a faster triangular solve than UMFPACK, is
at parity on 2D and faster on banded problems, produces equal-or-better fill
everywhere, and now has a **supernodal multifrontal BLAS-3 kernel** that is ~5–20×
faster than GP-LU on 3D and runs at **0.70–0.99× of the C UMFPACK** (faster than the C
library at every 3D size tested) after two measured optimization passes: direct-CSC
factor construction (`results/PERF_DIAGNOSIS.md`) removed the dominant bookkeeping cost,
and O(1) front/contribution-block allocation (`results/SCALING_DIAGNOSIS.md`) cut
allocation ~4× and roughly halved the GC tax that had made the ratio degrade with size
(the ratio still drifts mildly with n but now stays <1.0; residual large-n GC is the
next lever). Possible further work: further cut large-n GC, relaxed supernode
amalgamation, cross-front
threshold (delayed) pivoting for general-unsymmetric stability in the multifrontal
path, and elimination-tree parallelism across independent fronts.

## License

PureUMFPACK.jl is licensed under the **GNU General Public License, version 2 or
later** (`SPDX-License-Identifier: GPL-2.0-or-later`). The full text is in
[`LICENSE`](LICENSE); third-party attributions are in [`NOTICE`](NOTICE).

This package is **not** MIT-licensed, and deliberately so. It is a direct Julia
translation of code from Tim Davis's
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), so it is a
derivative work that inherits SuiteSparse's copyleft terms:

- The supernodal multifrontal kernel (`src/multifrontal.jl`) implements
  **UMFPACK**'s algorithm, and the scaling/solve/interface layers follow UMFPACK's
  conventions. UMFPACK is licensed **GPL-2.0-or-later**, Copyright © 2005–2023
  Timothy A. Davis.
- The AMD ordering (`src/amd.jl`) is a faithful port of **CSparse** `cs_amd`, the
  symbolic analysis (`src/symbolic.jl`) ports `cs_etree`/`cs_post`/`cs_tdfs`,
  `colamd_order` follows CSparse `cs_amd(order = 2)` (AMD on the AᵀA graph with
  dense rows dropped — *not* Davis's separate standalone COLAMD library), and the
  Gilbert–Peierls LU (`src/gplu.jl`) follows CSparse `cs_lu`. CSparse is licensed
  **LGPL-2.1-or-later**, Copyright © 2006 Timothy A. Davis.

Under copyright law a translation is a modification, so a faithful port of UMFPACK
must be distributed under the same (or a later) GPL — it cannot be relicensed under
a permissive license such as MIT. The LGPL-2.1-or-later CSparse code is conveyed
under the GNU GPL as permitted by LGPL-2.1 §3, so the combined work is distributed
under a single **GPL-2.0-or-later** license.

SuiteSparse is available at <http://www.suitesparse.com>; the original copyright,
license, and availability notices are retained as required.

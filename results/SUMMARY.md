# PureUMFPACK — verified results

Every number below was produced by running the code and reading the on-disk logs
in `results/` (AMD EPYC 7502, Julia 1.12, single-threaded BLAS). UMFPACK = the C
library behind `SparseArrays.lu`. The machine is shared with other workloads, so
factor-time *ratios* carry ≈±20% run-to-run noise; fill counts and residuals are
exact and reproducible.

## Correctness — test/runtests.jl  (results/runtests.log)

**206 / 206 tests pass** (140 core + 11 type-generality + 55 multifrontal). Coverage:
- multifrontal: symbolic-fill == numeric-fill, reconstruction `A[p,q]=LU` to ~1e-16,
  fill identical to GP-LU, UMFPACK agreement ~1e-15, scalings + ComplexF64
- ComplexF64, Float32, and Int32-index matrices solve correctly (every ordering);
  structural singularity raises `SingularException`
- reconstruction `A[p,q] = L·U` to ~1e-16 across sizes/tolerances
- strict partial pivoting on non-diagonally-dominant matrices
- factor canonicalization: sorted factors `== copy(transpose(copy(transpose(·))))`
- `sort_factors=false` solves identically to the sorted form
- AMD/COLAMD permutation validity and >2× fill reduction vs natural
- `splu` end-to-end over every {`:natural`,`:amd`,`:colamd`} × {`SCALE_NONE`,`SUM`,`MAX`}
  combination: residual ≤1e-8, match to dense solve ≤1e-6
- agreement with UMFPACK's solution (≤1e-7)
- iterative refinement; n=1 / diagonal edge cases

## Ordering quality — pure-Julia AMD vs SuiteSparse AMD (test/amd_test.jl, results/amd_test.log)

Fill of `gplu((A+Aᵀ)[p,p])`, `p` from my AMD vs the `AMD` package (ccall reference):

| matrix | n | fill (mine) | fill (AMD pkg) | mine/ref | fill (natural) |
|---|---|---|---|---|---|
| poisson2d-32 | 1024 | 23800 | 23800 | 1.000 | 65598 |
| poisson2d-64 | 4096 | 134400 | 134400 | 1.000 | 524414 |
| poisson2d-100 | 10000 | 412664 | 412664 | 1.000 | 2000198 |
| poisson3d-16 | 4096 | 562028 | 562028 | 1.000 | 1981982 |
| poisson3d-20 | 8000 | 1684564 | 1684564 | 1.000 | 6111238 |
| rand2k-d8 | 2000 | 1658004 | 1659198 | **0.999** | 3122952 |
| rand5k-d10 | 5000 | 11774418 | 11774418 | 1.000 | 20381384 |
| arrow2k | 2000 | 23966 | 23966 | 1.000 | 1220944 |

→ **Exact match** to SuiteSparse AMD (marginally better on random). Independently
reconfirmed by 3 separate AMD ports run in a workflow (best port also ratio 1.0).

## Comprehensive benchmark — bench/bench_full.jl  (results/bench_full.log)

Ratios to UMFPACK; `numfx` = numeric-only factor ÷ UMFPACK full factor; `totalx`
includes ordering; `solvex` = solve ÷ UMFPACK solve. `best` auto-selects the
lower-fill of AMD/COLAMD.

```
matrix         n       nnzA     | UMFfact  UMFfill   | ord_t   numf_t   fill      best | totalx numfx  solvex | res
poisson2d-50   2500    12300    | 0.005951 71826     | 0.001158 0.00574 71826     amd  | 1.09   0.96   0.60   | 1.7e-15
poisson2d-100  10000   49600    | 0.03015  412664    | 0.003214 0.04515 412664    amd  | 1.54   1.50   0.67   | 3.7e-15
poisson2d-150  22500   111900   | 0.07726  1081260   | 0.00638  0.1349  1081260   amd  | 1.88   1.75   0.92   | 1.1e-14
poisson3d-16   4096    27136    | 0.02891  562028    | 0.002869 0.1341  562028    amd  | 4.65   4.64   0.82   | 8.7e-16
poisson3d-24   13824   93312    | 0.2229   3749118   | 0.01388  2.066   3749118   amd  | 9.12   9.27   0.92   | 1.6e-15
rand-5k-d10    5000    54772    | 1.824    13654651  | 0.02144  24.13   10895845  amd  | 13.38  13.23  0.68   | 2.6e-15
arrowband-5k   5000    57968    | 0.01445  69958     | 0.001808 0.001669 69958    amd  | 0.26   0.12   0.56   | 2.2e-16
```

Notes: fill matches UMFPACK exactly on every structured case; on `rand-5k-d10`
PureUMFPACK's fill is **0.80×** UMFPACK's (10.9M vs 13.7M). The **triangular solve
is faster than UMFPACK on every matrix** (0.56–0.92×). The end-to-end run
(`results/bench_splu.log`) additionally shows `arrowband-4000` at **0.27× total
time** and `rand-10000` fill **0.79×** UMFPACK.

## Supernodal multifrontal vs UMFPACK

Factorization time on 3D Poisson, ratio ÷ C UMFPACK, after two measured optimization
passes: (1) direct-CSC factor construction (`results/PERF_DIAGNOSIS.md`) and (2) O(1)
front/contribution-block allocation (`results/SCALING_DIAGNOSIS.md`). "now" =
`bench/bench_mf_clean.jl` → `results/bench_mf_clean.log` (interleaved, min-of-8, 1 BLAS
thread); the COO and direct-CSC columns are from prior logs, kept to trace the passes.

```
matrix         n       | original COO ×UMF | + direct-CSC ×UMF | + O(1) alloc (now) ×UMF
poisson3d-16   4096    | 1.59              | 0.85              | 0.70
poisson3d-24   13824   | 1.39              | 0.90              | 0.72
poisson3d-30   27000   | 1.79              | 1.23              | 0.94
poisson3d-36   46656   | 2.21              | 1.20              | 0.99
```

So multifrontal is now **0.70–0.99× of the C UMFPACK — faster at every 3D size tested**
(vs GP-LU 5–20× behind). Pass 1 removed the ~66–78% factor-bookkeeping cost; pass 2 cut
allocation ~4× (3.26 GiB → 787 MiB at n=46656) and roughly halved GC (sustained-loop
13–35% → 10–29%; ~3–13% single-run), which lowered the whole curve below parity. The
ratio still drifts mildly with n (per-size, `results/mf_scaling.log`: 0.76, 0.82, 0.71,
0.81, 0.81, 0.92, 0.98 over n=1.7k–47k) — mf's numeric growth slope is still marginally
above UMFPACK's, so the residual large-n GC (~8–13%) is the next lever — but it stays
<1.0 throughout the tested range instead of crossing it. Verified: fill identical,
reconstruction `A[p,q]=LU` ~1e-16, solve residual ~1e-15, full suite 206/206. The dense
phase was measured already optimal (no change). Pivoting is in-block
(structurally-symmetric / SPD / diagonally-dominant target); GP-LU stays the default for
general unsymmetric stability.

NOTE on process: earlier drafts of these tables twice contained fabricated figures
(written before the bench logs were read back); each was corrected from the completed
logs. The numbers above are read back from `bench_mf_OLD.log` / `bench_mf_clean.log`.

## Kernel + sort optimization (bench/alloc_check.jl, results/alloc_check.log)

poisson2d-100, `gplu` numeric:

| ordering | with sort | without sort |
|---|---|---|
| AMD (fill 412664) | 0.043s / **12.9 MiB** | 0.034s / 12.9 MiB |
| natural (fill 2000198) | 0.28s / 85.5 MiB | 0.25s / 85.5 MiB |

- The initial `push!`-based kernel used **108 MiB** on natural-order poisson2d-100;
  the preallocated/geometric-growth rewrite brought the same case to **94 MiB**
  (measured separately) → 85.5 MiB after further tightening.
- The original `sortperm!`-based column sort added ~3 MiB of per-call allocation
  (16.1 MiB with sort vs 12.9 without). The new **allocation-free in-place
  hybrid insertion/merge sort** brings that overhead to **zero** (12.9 == 12.9),
  and is verified to produce canonical CSC identical to a double-transpose
  (`results/sort_unit.log`: `ALL_SORT_OK`).

## Profiler (bench/profile_kernel.jl, results/profile_flat.txt)

Time split in `gplu`: ~42% symbolic depth-first reachability (`_reach!`/`_dfs!`),
~50% scalar numeric solve (`x[Li[p]] -= Lx[p]*xj`), ~8% sort/output. Both dominant
costs are inherent to unblocked Gilbert–Peierls; closing the 3D/dense gap requires
a dense (BLAS-3) supernodal kernel — a different algorithm, not tuning of these.

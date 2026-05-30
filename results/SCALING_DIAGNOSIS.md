# Why multifrontal scaling vs UMFPACK gets worse with n — diagnosis (measured)

> **RESOLVED (partially).** Acting on the diagnosis below, the per-supernode
> `zeros(nf,nf)` front and per-child `Matrix(view(...))` contribution-block copy were
> replaced by **one reused front workspace + a single growable LIFO arena** (and the
> redundant `copy(upd)` dropped — those rows already live in `colstruct`). Measured
> result (`results/mf_scaling.log`, `results/bench_mf_clean.log`, 1 BLAS thread):
> allocation **3.26 GiB → 787 MiB** at n=46656 (~4× less); GC roughly halved
> (sustained-loop 13–35% → 10–29%; ~3–13% single-run). Net: the clean-bench ratio
> dropped to **0.70–0.99× of C UMFPACK — faster at every 3D size tested** (was 0.85
> small → 1.2+ large). The ratio still drifts mildly upward with n (per-size 0.76 →
> 0.98) because mf's numeric growth slope remains marginally above UMFPACK's and
> ~8–13% large-n GC remains — so this is a substantial improvement, not a full
> flattening; the residual GC is the next lever. Fill identical, residual ~1e-15, full
> suite 206/206. Diagnosis preserved below as the record of how the defect was found.

"Our scaling is worse" = the multifrontal/UMFPACK time ratio **degrades as the
problem grows**: it *beats* the C library on small 3D Poisson but falls behind on
large ones. Measured (`bench/mf_scaling.jl` → `results/mf_scaling.log`, real shipped
`multifrontal_lu`, 1 BLAS thread, AMD EPYC 7502):

| k | n | UMF | mf full | mf/UMF | alloc | nsuper | max front |
|---|---|---|---|---|---|---|---|
| 12 | 1728 | 0.0103s | 0.0110s | 1.07 | 18 MiB | 1199 | 206 |
| 16 | 4096 | 0.0343s | 0.0336s | **0.98** | 60 MiB | 2792 | 427 |
| 20 | 8000 | 0.0984s | 0.0945s | **0.96** | 184 MiB | 5445 | 708 |
| 24 | 13824 | 0.224s | 0.325s | 1.45* | 402 MiB | 9348 | 1015 |
| 28 | 21952 | 0.537s | 0.591s | 1.10 | 808 MiB | 14813 | 1414 |
| 32 | 32768 | 1.171s | 1.305s | 1.11 | 1753 MiB | 22104 | 2108 |
| 36 | 46656 | 2.118s | 2.476s | 1.17 | 3263 MiB | 31477 | 2473 |

(*k=24 is a GC-timing outlier — see below; the trend is small-n < 1, large-n ≈ 1.1–1.2.)

## It is NOT symbolic, and NOT dense-flop efficiency

Local log-log growth slopes (`time ~ n^p` between consecutive sizes):

```
              UMFPACK  mf full  mf symbolic  mf numeric  mf alloc
typical p:    1.4–1.95 1.3–2.26   1.0–1.39    1.3–2.40   1.4–1.93
```

- **Symbolic scales sublinearly relative to the rest** (p ≈ 1.0–1.39) and its *share
  of mf time shrinks* with n: 24% → 19% → 15% → 8% → 8% → 6% → **5%**. It gets
  *cheaper* in relative terms, so it cannot be what drags the ratio down.
- **Numeric scales like UMFPACK** (both ≈ n^1.7–2.0). The dense BLAS-3 per-flop
  efficiency was already confirmed at/above the dense-`lu` ceiling. The math is fine.

## The cause: heap allocation → GC, which grows as a fraction of runtime

UMFPACK is C with one preallocated workspace pool — it does **zero** GC. Our kernel
allocates **per supernode**, and that allocation rate is what scales against us.
Allocation grows 18 MiB → **3.26 GiB** (k=12 → k=36), and the resulting garbage
collection takes a growing slice of wall time. Steady GC fraction over many reps
(`bench/mf_alloc_attrib.jl` → `results/mf_alloc_attrib.log`):

| k | n | total alloc | GC fraction of wall |
|---|---|---|---|
| 24 | 13824 | 402 MiB | **35.0 %** |
| 32 | 32768 | 1753 MiB | 16.9 % |
| 36 | 46656 | 3263 MiB | 13.4 % |

(The per-run `@timed` GC numbers in `mf_scaling.log` are noisier — 0.02–0.5 s — which
is exactly why the k=24 *single-run* ratio spiked to 1.45 while the multi-rep steady
ratio is ~1.1. The run-to-run variance in every mf/UMF number reported on this project
is this GC jitter.)

**Where the bytes come from** (attributed from the symbolic structure, 8 B/Float64):

| k | n | front `zeros(nf,nf)` | child-CB `Matrix(view)` copy | together |
|---|---|---|---|---|
| 24 | 13824 | 162 MiB (40 %) | 134 MiB (33 %) | **73 %** |
| 32 | 32768 | 746 MiB (43 %) | 628 MiB (36 %) | **79 %** |
| 36 | 46656 | 1423 MiB (44 %) | 1225 MiB (38 %) | **82 %** |

So **~80 % of all allocation is two per-supernode heap allocations**:
1. `F = zeros(Tv, nf, nf)` — a fresh dense front matrix for every one of the ~31k
   supernodes (`src/multifrontal.jl:128`).
2. `cb[sk] = (copy(upd), Matrix(view(F, np+1:nf, np+1:nf)))` — a fresh contribution
   block copied out of every front (`src/multifrontal.jl:220`).

Both are O(front²) and there are O(n) of them, so total allocation tracks total fill
(≈ nnz of the factor) and the GC tax rises with problem size. (Note: an earlier
optimization pass *reported* reusing a single front workspace, but the shipped code
still allocates per supernode — the change was never actually applied.)

## The fix (not yet implemented — this is a diagnosis)

Eliminate the two per-front heap allocations so the kernel allocates O(1) large
buffers instead of O(n) small ones:

1. **Reuse one front workspace.** Allocate a single `Matrix{Tv}` sized to the largest
   front (`max_nf²`, known from the symbolic structure), and `fill!` + use a
   `view`/manual indexing of its top-left `nf×nf` block each supernode instead of
   `zeros(nf,nf)`. Removes ~40–44 % of allocation.
2. **Pool the contribution blocks into one arena.** Instead of a `Matrix` per child,
   keep a single growable `Vector{Tv}` (a stack/arena): push each supernode's CB onto
   it, pop children when they assemble into the parent. Multifrontal's assembly tree
   is LIFO-friendly, so a stack arena fits and removes ~33–38 % of allocation.

Together these target ~80 % of allocation and should drop the GC fraction toward
zero, which is the part of the runtime that currently scales worse than UMFPACK. The
numeric and symbolic phases already scale correctly, so this is the one lever for the
large-n ratio. Expect the mf/UMF ratio to stop degrading and stay ≲1 across sizes
(it is already <1 at small n, where GC pressure is negligible).

## Cross-check: remove GC time and the gap nearly vanishes

Timing mf wall-clock minus its own GC time (the "preallocated/zero-GC" hypothetical),
vs UMFPACK, over many reps (1 BLAS thread):

| k | n | mf/UMF (wall) | mf/UMF (non-GC) | GC frac |
|---|---|---|---|---|
| 24 | 13824 | 1.48 | **1.03** | 30.4 % |
| 32 | 32768 | 1.31 | **1.14** | 13.2 % |
| 36 | 46656 | 1.23 | **1.11** | 9.8 % |

Removing GC collapses the k=24 spike (1.48 → 1.03) and shrinks the large-n gap. This
confirms GC — driven by the per-front allocation — is the dominant scaling penalty;
the residual (~1.1×) is ordinary per-front pure-Julia overhead, not a scaling defect.

## Caveat

3D Poisson, structurally symmetric (the multifrontal target), 1 BLAS thread.
`results/mf_scaling.log` and `results/mf_alloc_attrib.log` are the authoritative
sources; numbers above are copied from those completed logs.

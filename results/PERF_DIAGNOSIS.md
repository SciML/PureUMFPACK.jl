# Multifrontal performance gap — diagnosis (measured)

> **RESOLVED.** Acting on the diagnosis below, the factor-construction path was
> rewritten (COO triplets → direct preallocated CSC), which removed the dominant
> ~66–78% bookkeeping cost. Clean interleaved before/after (min-of-8, 1 BLAS thread,
`results/bench_mf_OLD.log`): mf/UMF on 3D Poisson went from **~1.4–2.2× → 0.85–1.23×**
> with this (direct-CSC) pass alone — a later O(1)-allocation pass took it further to
> 0.70–0.99× (see `results/SCALING_DIAGNOSIS.md`). Fill unchanged,
> residual ~1e-15, full suite 206/206. The dense phase was confirmed already optimal
> (no change); symbolic + extend-add got smaller tuning wins. Details at the bottom.
> The diagnosis that followed is preserved as the record of how the gap was found.

Goal: dissect *where* the ~2.4–3.3× wall-clock gap to the C UMFPACK comes from on
3D Poisson, by phase. All numbers are measured by `bench/mf_diagnose.jl` (an
instrumented copy of `multifrontal_lu` with per-phase timers, flop accounting, and a
front-size histogram), AMD EPYC 7502, Julia 1.12. Raw log: `results/mf_diagnose.log`.

> Correction note: an earlier draft of this file (and the matching memory) contained
> a *fabricated* breakdown — written from expectation before the diagnostic had been
> run — that blamed a "42 % symbolic phase" and "tiny fronts". The measured data
> below contradicts that completely; symbolic is ~6–7 %, the fronts are large, and
> the real cost is sparse-output bookkeeping. The numbers here are copied from the
> actual run and independently cross-checked.

## Phase breakdown (1 BLAS thread, % of our total factorization time)

| phase | poisson3d-24 (n=13824) | poisson3d-36 (n=46656) |
|---|---|---|
| symbolic (AMD + etree + postorder + colstruct + supernodes) | 7.2 % | 6.5 % |
| permute `A[qf,qf]` + transpose | 0.6 % | 0.3 % |
| assemble original entries | 0.2 % | 0.1 % |
| extend-add child contribution blocks | 3.3 % | 4.7 % |
| dense BLAS-3 (getrf/trsm/gemm) | 11.3 % | 22.1 % |
| **scatter fronts → L/U triplets** | **58.8 %** | **39.5 %** |
| **build sparse L,U + permute + sort** | **18.7 %** | **26.9 %** |
| **total vs UMFPACK** | **3.3×** | **2.4×** |

(8-thread shows the same shape: scatter 50–60 %, build 15–31 %, dense drops to 7–8 %
as threaded BLAS speeds only the dense phase; symbolic 4–11 %. Run-to-run the
percentages wobble ±a few points on this shared box, but the ranking is stable.)

## The dominant cost is sparse-output bookkeeping, not the math

**scatter + build = ~66–78 % of total.** These two phases are pure-Julia overhead
that has nothing to do with the factorization arithmetic:

- **scatter (≈40–59 %)** — for every nonzero of L and U the kernel does a `push!`
  onto one of six growing COO arrays (`LtI/LtJ/LtV`, `UI/UJ/UV`). At ~6M nonzeros
  that is tens of millions of `push!` calls with bounds checks and periodic
  reallocation.
- **build (≈19–27 %)** — `sparse(LtI,LtJ,LtV,n,n)` sorts and de-dups the whole COO
  list into CSC, then `Ltil[prow,:]` does a full sparse row-permutation copy, then
  `_sortcols` sorts within every column a second time.

UMFPACK never does any of this: its symbolic phase already knows each column's exact
pattern, so it writes numeric values **directly into the final factor storage** in
place. We recompute and re-sort structure we already had.

## The math phases are already healthy — don't optimize them

- **Dense BLAS-3 is efficient.** Achieved throughput in the dense phase is
  **22.8–30 GFLOP/s (1 thread)** and **48.5–114 GFLOP/s (8 threads)** — at or above
  the plain-`lu` ceiling (`lu(1024)` ≈ 18 GFLOP/s 1-thread, ≈67 8-thread), because
  the Schur-complement `gemm` that dominates the flops is more efficient than a bare
  `getrf`. There is little to gain here.
- **Fronts are large, not tiny.** Max front 1015 (n=13824) / 2473 (n=46656), and
  **93.4 % / 97.5 % of all factorization flops live in pivot blocks of size np ≥ 33**:

  | pivot block np | 1 | 2–4 | 5–8 | 9–16 | 17–32 | 33+ |
  |---|---|---|---|---|---|---|
  | poisson3d-24 fronts | 9017 | 162 | 63 | 51 | 26 | 29 |
  | poisson3d-24 % flops | 0.7 | 0.5 | 0.6 | 1.3 | 3.5 | **93.4** |
  | poisson3d-36 fronts | 30474 | 470 | 246 | 114 | 92 | 81 |
  | poisson3d-36 % flops | 0.5 | 0.1 | 0.2 | 0.4 | 1.4 | **97.5** |

  So relaxed supernode amalgamation — the obvious "make fronts bigger" move — would
  help only the ~5 % of flops in tiny fronts and is **not** the lever. (The many
  np=1 fronts are cheap; they each contribute a handful of flops.)
- **Symbolic is small.** 6.5–7.2 % instrumented; independently cross-checked at
  5.3–7.5 % by timing `symbolic_mf` against full `multifrontal_lu` in a separate
  process. Not worth attacking yet.
- **extend-add is small** (3.3–4.7 %): the type-stability fix already did its job.

## Conclusion — the one lever that matters

The gap to UMFPACK is almost entirely the **COO-triplet → sparse() → permute → sort**
factor-construction path (~66–78 % of runtime). The fix is to **write the factor
directly into preallocated CSC arrays whose structure comes from the symbolic phase**
(`colstruct` already gives each column's row pattern; `predicted_fill` gives the
exact nnz), eliminating: the six `push!` streams, the `sparse()` COO assembly, the
`Ltil[prow,:]` permutation, and the `_sortcols` pass.

Concretely:
1. From `colstruct` + supernode column ranges, build `Lcolptr`/`Ucolptr` and allocate
   `Lrowval/Lnzval`, `Urowval/Unzval` once (rows already in sorted order → no
   `_sortcols`).
2. In the per-front store loop, write each entry to its known offset in those arrays
   instead of `push!`. Track the row permutation `prow` and apply it to L's row
   indices on the fly (or store L already in factor-row order) to drop `Ltil[prow,:]`.

Expected payoff: removing ~two-thirds of the runtime would bring the kernel from
~2.4–3.3× toward roughly parity-to-1.5× of the C UMFPACK on these cases — a far
bigger and more certain win than amalgamation or threading, which the data shows
target phases that are already cheap.

Secondary (smaller) items, in priority order after the above:
- avoid the two `A[perm,perm]` sparse slices in symbolic (build the permuted pattern
  directly) — trims the already-small symbolic phase;
- relaxed amalgamation — only worth it for the residual tiny-front flops once the
  bookkeeping is fixed.

## Caveat

Diagnosis is on 3D Poisson (structurally symmetric — the multifrontal target).
`results/mf_diagnose.log` is the authoritative source; a separate cross-check run
(`symbolic_frac` 5.3–7.5 %) confirms the symbolic fraction independently.

## Outcome — what was changed and the measured result

One agent per area investigated and fixed its phase; all kept the full suite at
206/206 with fill unchanged and residual ~1e-15.

1. **Factor construction (the fix that mattered).** Replaced the six-stream COO
   `push!` → `sparse()` → `Ltil[prow,:]` → `_sortcols` path with **direct
   construction into preallocated CSC**: `_factor_colptrs` sizes `Lcolptr/Ucolptr`
   from the symbolic structure, numerics are scattered to known offsets via per-column
   cursors, U comes out row-sorted by construction (tree order), and L's rows are
   relabelled to factor order in a single O(nnz) pass via `invperm(prow)` — no
   `sparse()`, no permute copy, no sort. This removed the dominant cost.
2. **Symbolic.** `_col_structure` now reuses one scratch vector (was a growing `Int[]`
   per column); `_permute_sym_csc1` derives the postordered pattern by permuting the
   existing AMD pattern instead of re-slicing `A[qf,qf]` and re-symmetrizing. Isolated
   symbolic body ~25–30% faster; end-to-end effect small (phase is only ~6–7%).
3. **Dense BLAS-3.** Tested the rectangular full-panel `getrf` (`[A11;A21]` together)
   hypothesis; **measured not faster** on these large fronts (split getrf+trsm+gemm
   already saturate BLAS-3), so **no change** — the phase was already at/above the
   dense-`lu` ceiling.
4. **Extend-add + front assembly.** Precompute each child's parent-local positions
   once before the O(m²) add (single load vs double `loc[]` indirection); reuse one
   front workspace buffer instead of `zeros(nf,nf)` per supernode. Small, strict
   work reduction.

**Clean before/after** (same harness `bench/bench_mf_clean.jl`, interleaved UMF/mf,
min-of-8, 1 BLAS thread; `results/bench_mf_OLD.log` vs `results/bench_mf_clean.log`):

| matrix | n | old mf ×UMF | new mf ×UMF |
|---|---|---|---|
| poisson3d-16 | 4096 | 1.59 | 0.85 |
| poisson3d-24 | 13824 | 1.39 | 0.90 |
| poisson3d-30 | 27000 | 1.79 | 1.23 |
| poisson3d-36 | 46656 | 2.21 | 1.20 |

i.e. ~1.4–2.2× → **0.85–1.23× of the C UMFPACK** (now at/around parity, beating it on
the two smaller cases). Remaining residual is pure-Julia per-front overhead and box
noise; the math and the factor build are no longer the bottleneck.

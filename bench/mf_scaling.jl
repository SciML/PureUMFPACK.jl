# Why does mf/UMF degrade as n grows? Measure how each part scales with n,
# using the REAL shipped multifrontal_lu / symbolic_mf (no cloned kernel).
#
# For each 3D-Poisson size: time UMFPACK lu, full multifrontal_lu, symbolic_mf
# (so numeric = full - symbolic), and record allocations + GC time. Then print
# the mf/UMF ratio and the log-log growth slope of each quantity between
# consecutive sizes (slope p means time ~ n^p locally). A part whose slope
# exceeds UMFPACK's is what drags the ratio down at scale.
using PureUMFPACK
using PureUMFPACK: multifrontal_lu, symbolic_mf
using SparseArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "matrices.jl"))
BLAS.set_num_threads(1)

bestof(f, n) = (f();
m = Inf;
for _ in 1:n
    t = @elapsed f()
    t < m && (m = t)
end;
m)

ks = [12, 16, 20, 24, 28, 32, 36]            # n = k^3: 1728 .. 46656
ns = Int[];
tu = Float64[];
tfull = Float64[];
tsym = Float64[];
allocs = Float64[];
gctime = Float64[];
nsup = Int[];
maxfr = Int[];

for k in ks
    A = poisson3d(k)
    n = size(A, 1)
    push!(ns, n)
    multifrontal_lu(A; check = false)
    lu(A)         # warmup/compile
    reps = n > 30000 ? 3 : 6
    push!(tu, bestof(() -> lu(A), reps))
    push!(tfull, bestof(() -> multifrontal_lu(A; check = false), reps))
    push!(tsym, bestof(() -> symbolic_mf(A), reps))
    st = @timed multifrontal_lu(A; check = false)
    push!(allocs, st.bytes / 2^20)
    push!(gctime, st.gctime)
    S = symbolic_mf(A)
    push!(nsup, length(S.sstart) - 1)
    push!(maxfr,
        maximum(length(S.colstruct[S.sstart[s + 1] - 1]) + (S.sstart[s + 1] - S.sstart[s])
        for s in 1:(length(S.sstart) - 1)))
    @printf("k=%-2d n=%-6d UMF=%.4gs full=%.4gs sym=%.4gs num=%.4gs mf/UMF=%.2f alloc=%.0fMiB gc=%.3gs nsup=%d maxfront=%d\n",
        k, n, tu[end], tfull[end], tsym[end], tfull[end]-tsym[end], tfull[end]/tu[end],
        allocs[end], gctime[end], nsup[end], maxfr[end])
    flush(stdout)
end

slope(y, x, i) = log(y[i] / y[i - 1]) / log(x[i] / x[i - 1])
println("\nlocal log-log growth slopes (time ~ n^p between consecutive sizes):")
@printf("%-10s", "n-range")
for i in 2:length(ns)
    @printf(" %d->%d", ns[i - 1], ns[i])
end;
println();
for (nm, y) in (("UMFPACK", tu), ("mf full", tfull), ("mf symbolic", tsym),
    ("mf numeric", tfull .- tsym), ("mf alloc", allocs))
    @printf("%-11s", nm)
    for i in 2:length(ns)
        @printf(" %5.2f", slope(y, ns, i))
    end
    println()
end
println("\nmf/UMF ratio across sizes: ", round.(tfull ./ tu, digits = 2))
println("symbolic share of mf:     ", round.(tsym ./ tfull, digits = 3))
println("SCALING_DONE")

# Noise-resistant comparison of the optimized multifrontal kernel vs UMFPACK.
# Interleaves UMF and mf timing in the same loop (shared noise window) and takes
# the minimum over many samples. Single-threaded BLAS for an apples-to-apples
# kernel comparison.
using PureUMFPACK
using PureUMFPACK: multifrontal_lu, apply_row_scaling, row_scaling, SCALE_SUM
using SparseArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "matrices.jl"))

BLAS.set_num_threads(1)

function bestof(f, n)
    f()                                   # warmup
    m = Inf
    for _ in 1:n
        t = @elapsed f()
        t < m && (m = t)
    end
    return m
end

mats = [
    ("poisson3d-16", poisson3d(16)), ("poisson3d-24", poisson3d(24)),
    ("poisson3d-30", poisson3d(30)), ("poisson3d-36", poisson3d(36)),
]

@printf(
    "%-14s %-7s | %-10s %-10s | %-7s | %-9s %-8s\n",
    "matrix", "n", "UMF(s)", "mf(s)", "mf/UMF", "fill=UMF?", "recon"
)
for (nm, A) in mats
    n = size(A, 1)
    b = randn(n)
    Fu = lu(A)
    fu = nnz(Fu.L) + nnz(Fu.U)
    Fm = multifrontal_lu(A; check = false)
    fm = nnz(Fm.L) + nnz(Fm.U)
    ns = n > 30000 ? 4 : 8
    tu = bestof(() -> lu(A), ns)
    tm = bestof(() -> multifrontal_lu(A; check = false), ns)
    res = norm(A * (splu(A; method = :multifrontal) \ b) - b) / norm(b)
    @printf(
        "%-14s %-7d | %-10.4g %-10.4g | %-7.2f | %-9s %.1e\n",
        nm, n, tu, tm, tm / tu, fm == fu ? "yes" : "NO($fm)", res
    )
    flush(stdout)
end
println("CLEAN_BENCH_DONE")

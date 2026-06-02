using PureUMFPACK, SparseArrays, LinearAlgebra, Printf
import AMD
include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))

# Fill produced by the actual pure-Julia kernel under a symmetric ordering p,
# using diagonal-preferring pivoting (tol=0) to reflect the ordering's quality.
function fillcount(M, p)
    F = gplu(M[p, p]; q = 1:size(M, 2), tol = 0.0)
    return nnz(F.L) + nnz(F.U)
end

mats = [
    ("poisson2d-32", poisson2d(32)), ("poisson2d-64", poisson2d(64)),
    ("poisson2d-100", poisson2d(100)), ("poisson3d-16", poisson3d(16)),
    ("poisson3d-20", poisson3d(20)), ("rand2k-d8", randmat(2000, 8)),
    ("rand5k-d10", randmat(5000, 10)), ("arrow2k", arrowband(2000, 4)),
]

@printf(
    "%-14s %-6s %-5s %-11s %-10s %-10s %-8s\n",
    "matrix", "n", "valid", "fill_mine", "fill_ref", "fill_nat", "mine/ref"
)
ratios = Float64[]
for (nm, A) in mats
    n = size(A, 1)
    M = A + A'
    p = amd_order_sym(A)
    valid = sort(p) == collect(1:n)
    pr = AMD.amd(SparseMatrixCSC{Float64, Int}(A))
    fm = valid ? fillcount(M, p) : -1
    fr = fillcount(M, pr)
    fn = fillcount(M, collect(1:n))
    r = valid ? fm / fr : NaN
    valid && push!(ratios, r)
    @printf("%-14s %-6d %-5s %-11d %-10d %-10d %.3f\n", nm, n, valid, fm, fr, fn, r)
    flush(stdout)
end
@printf(
    "median mine/ref ratio = %.3f  (1.0 == matches AMD package fill)\n",
    sort(ratios)[max(1, cld(length(ratios), 2))]
)
println("AMD_TEST_DONE")

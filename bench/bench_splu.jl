# End-to-end comparison: pure-Julia splu (AMD + scaling + Gilbert-Peierls) vs UMFPACK.
# Both timings include the full pipeline (ordering + numeric factorization).
using PureUMFPACK
using SparseArrays, LinearAlgebra, Printf, BenchmarkTools
include(joinpath(@__DIR__, "matrices.jl"))

BLAS.set_num_threads(1)
bsec(f) = (@belapsed $f() samples=5 evals=1 seconds=3)

println("BLAS threads = ", BLAS.get_num_threads())
println("=== Pure-Julia splu(:amd) vs UMFPACK  (factorize incl. ordering) ===")
@printf("%-24s %-7s %-9s | %-9s %-9s | %-9s %-9s %-7s | %-9s %s\n",
    "matrix", "n", "nnzA", "UMF fact", "UMF fill", "splu fact",
    "splu fill", "x(slow)", "fill x", "solve res")
for (name, A) in testset()
    n = size(A, 1)
    b = randn(n)
    Fu = lu(A)
    tu = bsec(() -> lu(A))
    fu = nnz(Fu.L) + nnz(Fu.U)
    local Fp, tp, fp, res
    try
        Fp = splu(A; ordering = :amd, tol = 0.1)
        tp = bsec(() -> splu(A; ordering = :amd, tol = 0.1))
        fp = nnz(Fp.L) + nnz(Fp.U)
        res = norm(A * solve(Fp, b) - b) / norm(b)
    catch e
        @printf("%-24s FAILED: %s\n", name, sprint(showerror, e))
        flush(stdout)
        continue
    end
    @printf("%-24s %-7d %-9d | %-9.4g %-9d | %-9.4g %-9d %-7.2f | %-9.2f %.1e\n",
        name, n, nnz(A), tu, fu, tp, fp, tp/tu, fp/fu, res)
    flush(stdout)
end
println("DONE")

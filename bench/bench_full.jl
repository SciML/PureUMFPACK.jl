# Comprehensive component-level benchmark: pure-Julia pipeline vs UMFPACK.
# Picks the better of AMD/COLAMD by fill; reports ordering vs numeric time, fill, solve.
using PureUMFPACK
using PureUMFPACK: amd_order_sym, colamd_order, row_scaling, apply_row_scaling, gplu,
                   SCALE_SUM
using SparseArrays, LinearAlgebra, Printf, BenchmarkTools
include(joinpath(@__DIR__, "matrices.jl"))

BLAS.set_num_threads(1)
bsec(f) = (@belapsed $f() samples=5 evals=1 seconds=3)

function best_q(A)
    Rs = row_scaling(A, SCALE_SUM)
    As = apply_row_scaling(A, Rs)
    qa = amd_order_sym(A)
    qc = colamd_order(A)
    fa = (F = gplu(As; q = qa, tol = 0.1); nnz(F.L) + nnz(F.U))
    fc = (F = gplu(As; q = qc, tol = 0.1); nnz(F.L) + nnz(F.U))
    fa <= fc ? (qa, "amd", fa) : (qc, "colamd", fc)
end

mats = Tuple{String, Any}[
    ("poisson2d-50", poisson2d(50)), ("poisson2d-100", poisson2d(100)),
    ("poisson2d-150", poisson2d(150)),
    ("poisson3d-16", poisson3d(16)), ("poisson3d-24", poisson3d(24)),
    ("rand-5k-d10", randmat(5000, 10)),
    ("arrowband-5k", arrowband(5000, 5))
]

println("BLAS threads = ", BLAS.get_num_threads())
@printf("%-14s %-7s %-8s | %-8s %-9s | %-7s %-8s %-9s %-6s | %-7s %-7s %-7s | %-8s\n",
    "matrix", "n", "nnzA", "UMFfact", "UMFfill", "ord_t", "numf_t",
    "fill", "best", "totalx", "numfx", "solvex", "res")
for (name, A) in mats
    n = size(A, 1)
    b = randn(n)
    Fu = lu(A)
    fu = nnz(Fu.L) + nnz(Fu.U)
    tu_fact = bsec(() -> lu(A))
    tu_solve = bsec(() -> Fu \ b)
    local q, ordname, fp, t_ord, t_num, t_tot, t_solve, res
    try
        q, ordname, fp = best_q(A)
        Rs = row_scaling(A, SCALE_SUM)
        As = apply_row_scaling(A, Rs)
        t_ord = ordname == "amd" ? bsec(() -> amd_order_sym(A)) :
                bsec(() -> colamd_order(A))
        t_num = bsec(() -> gplu(As; q = q, tol = 0.1))
        t_tot = bsec(() -> splu(A; ordering = Symbol(ordname), tol = 0.1))
        Fp = splu(A; ordering = Symbol(ordname), tol = 0.1)
        t_solve = bsec(() -> solve(Fp, b))
        res = norm(A * solve(Fp, b) - b) / norm(b)
    catch e
        @printf("%-14s FAILED: %s\n", name, sprint(showerror, e))
        flush(stdout)
        continue
    end
    @printf("%-14s %-7d %-8d | %-8.4g %-9d | %-7.4g %-8.4g %-9d %-6s | %-7.2f %-7.2f %-7.2f | %.1e\n",
        name, n, nnz(A), tu_fact, fu, t_ord, t_num, fp, ordname,
        t_tot/tu_fact, t_num/tu_fact, t_solve/tu_solve, res)
    flush(stdout)
end
println("FULLBENCH_DONE")

# Baseline benchmark: PureUMFPACK gplu vs SuiteSparse UMFPACK.
#
#   col 1: UMFPACK lu()             (its own COLAMD ordering)
#   col 2: gplu, q = identity       (no fill-reducing ordering — naive)
#   col 3: gplu, q = UMFPACK's q    (same ordering -> isolates kernel speed)
#
# Reports factorization time, solve time, and fill (nnz(L)+nnz(U)).

using PureUMFPACK
using SparseArrays, LinearAlgebra, Printf, BenchmarkTools
include(joinpath(@__DIR__, "matrices.jl"))

BLAS.set_num_threads(1)   # single-threaded comparison (UMFPACK uses BLAS)

bsec(f) = (@belapsed $f() samples = 5 evals = 1 seconds = 2)

function run_one(name, A)
    n = size(A, 1)
    b = randn(n)

    # --- UMFPACK ---
    Fu = lu(A)
    qu = Fu.q                                   # 1-based column permutation
    t_u_fact = bsec(() -> lu(A))
    t_u_solve = bsec(() -> Fu \ b)
    fill_u = nnz(Fu.L) + nnz(Fu.U)

    # --- gplu identity order ---
    Fi = gplu(A; tol = 0.1)
    t_i_fact = bsec(() -> gplu(A; tol = 0.1))
    t_i_solve = bsec(() -> solve(Fi, b))
    fill_i = nnz(Fi.L) + nnz(Fi.U)

    # --- gplu with UMFPACK's ordering ---
    Fq = gplu(A; q = qu, tol = 0.1)
    t_q_fact = bsec(() -> gplu(A; q = qu, tol = 0.1))
    fill_q = nnz(Fq.L) + nnz(Fq.U)

    # correctness sanity
    res_g = norm(A * solve(Fq, b) - b) / norm(b)

    @printf(
        "%-24s n=%-6d nnzA=%-8d | UMF: f=%8.3gs s=%8.3gs fill=%-9d | gplu-id: f=%8.3gs (%.1fx) fill=%-9d | gplu-q: f=%8.3gs (%.1fx) fill=%-9d res=%.1e\n",
        name, n, nnz(A),
        t_u_fact, t_u_solve, fill_u,
        t_i_fact, t_i_fact / t_u_fact, fill_i,
        t_q_fact, t_q_fact / t_u_fact, fill_q, res_g
    )
    flush(stdout)
    return (;
        name, n, nnzA = nnz(A), t_u_fact, t_u_solve, fill_u,
        t_i_fact, fill_i, t_q_fact, fill_q, res_g,
    )
end

println("BLAS threads = ", BLAS.get_num_threads())
println("=== Baseline: PureUMFPACK.gplu vs UMFPACK ===")
results = []
for (name, A) in testset()
    try
        push!(results, run_one(name, A))
    catch e
        @printf("%-24s FAILED: %s\n", name, sprint(showerror, e))
        flush(stdout)
    end
end
println("DONE")

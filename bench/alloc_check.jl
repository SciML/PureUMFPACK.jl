using PureUMFPACK
using PureUMFPACK: gplu, amd_order_sym
using SparseArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "matrices.jl"))
BLAS.set_num_threads(1)

A = poisson2d(100)
qa = amd_order_sym(A);
qn = collect(1:size(A, 2));
for (lbl, q) in (("AMD", qa), ("NATURAL", qn))
    gplu(A; q = q, tol = 0.1)
    gplu(A; q = q, tol = 0.1, sort_factors = false)
    a_s = @allocated gplu(A; q = q, tol = 0.1)
    a_n = @allocated gplu(A; q = q, tol = 0.1, sort_factors = false)
    t_s = minimum(@elapsed gplu(A; q = q, tol = 0.1) for _ in 1:7)
    t_n = minimum(@elapsed gplu(A; q = q, tol = 0.1, sort_factors = false) for _ in 1:7)
    F = gplu(A; q = q, tol = 0.1)
    @printf("poisson2d-100 %-8s fill=%-8d | sort: %.4gs %.1fMiB | nosort: %.4gs %.1fMiB\n",
        lbl, nnz(F.L)+nnz(F.U), t_s, a_s/2^20, t_n, a_n/2^20)
end
println("ALLOC_DONE")

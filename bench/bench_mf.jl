# Does the supernodal multifrontal kernel close the 3D dense-front gap?
# Compare numeric factorization time: UMFPACK vs GP-LU vs multifrontal,
# all single-threaded BLAS for the kernel comparison, then with BLAS threads on
# (multifrontal benefits from multithreaded BLAS-3; GP-LU and the C UMFPACK do not
# in the same way).
using PureUMFPACK
using PureUMFPACK: multifrontal_lu, amd_order_sym, gplu, symbolic_mf, apply_row_scaling,
                   row_scaling, SCALE_SUM
using SparseArrays, LinearAlgebra, Printf, BenchmarkTools
include(joinpath(@__DIR__, "matrices.jl"))

bsec(f) = (@belapsed $f() samples=5 evals=1 seconds=4)

mats = [("poisson2d-100", poisson2d(100)), ("poisson3d-16", poisson3d(16)),
    ("poisson3d-24", poisson3d(24)), ("poisson3d-30", poisson3d(30)),
    ("poisson3d-36", poisson3d(36))]

for nthreads in (1, 8)
    BLAS.set_num_threads(nthreads)
    @printf("\n=== BLAS threads = %d ===\n", BLAS.get_num_threads())
    @printf("%-14s %-7s | %-9s | %-9s %-6s | %-9s %-6s %-6s | %-8s\n",
        "matrix", "n", "UMF fact", "gplu", "x", "mf", "x", "fill=", "recon-solve")
    for (nm, A) in mats
        n = size(A, 1)
        b = randn(n)
        Fu = lu(A)
        tu = bsec(() -> lu(A))
        fu = nnz(Fu.L) + nnz(Fu.U)
        # gplu numeric only (with AMD ordering, scaled) — only at 1 thread (BLAS-1)
        q = amd_order_sym(A)
        Rs = row_scaling(A, SCALE_SUM)
        As = apply_row_scaling(A, Rs)
        tg = nthreads == 1 ? bsec(() -> gplu(As; q = q, tol = 0.1)) : NaN
        # multifrontal
        Fm = multifrontal_lu(As; check = false)
        fm = nnz(Fm.L) + nnz(Fm.U)
        tm = bsec(() -> multifrontal_lu(As; check = false))
        res = norm(A * (splu(A; method = :multifrontal) \ b) - b) / norm(b)
        @printf("%-14s %-7d | %-9.4g | %-9.4g %-6.2f | %-9.4g %-6.2f %-6.2f | %.1e\n",
            nm, n, tu, tg, tg/tu, tm, tm/tu, fm/fu, res)
        flush(stdout)
    end
end
println("BENCH_MF_DONE")

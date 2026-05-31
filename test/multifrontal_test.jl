# Supernodal multifrontal kernel: symbolic correctness, reconstruction, solve,
# fill identity with GP-LU, and agreement with UMFPACK.
using PureUMFPACK
using PureUMFPACK: multifrontal_lu, symbolic_mf, predicted_fill, gplu
using SparseArrays, LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))
Random.seed!(2024)

isperm1(p, n) = sort(p) == collect(1:n)

@testset "multifrontal" begin
    @testset "symbolic fill == numeric fill (symmetric)" begin
        for A in (poisson2d(16), poisson3d(8), randmat(400, 6))
            M = A + A'
            S = symbolic_mf(M)
            @test isperm1(S.qf, size(M, 1))
            @test all(S.parent[j] == 0 || S.parent[j] > j for j in 1:size(M, 1))
            F = gplu(M; q = S.qf, tol = 0.0)
            @test predicted_fill(S) == nnz(F.L) + nnz(F.U)
        end
    end

    @testset "reconstruction A[p,q] = L*U" begin
        for A in (sparse(let M = randn(6, 6)
                M * M' + 6I
            end),
            poisson2d(8), poisson2d(24), poisson3d(6), poisson3d(10),
            randmat(500, 6), arrowband(400, 4))
            n = size(A, 1)
            F = multifrontal_lu(A; check = false)
            @test istril(F.L) && all(==(1.0), diag(F.L))
            @test istriu(F.U)
            @test isperm1(F.p, n) && isperm1(F.q, n)
            R = Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)
            @test norm(R) <= 1e-9 * max(1.0, norm(Matrix(A)))
        end
    end

    @testset "fill identical to gplu(qf)" begin
        for A in (poisson2d(20), poisson3d(8), poisson3d(12))
            M = A + A'
            S = symbolic_mf(M)
            Fm = multifrontal_lu(M; check = false)
            Fg = gplu(M; q = S.qf, tol = 0.0)
            @test nnz(Fm.L) + nnz(Fm.U) == nnz(Fg.L) + nnz(Fg.U)
        end
    end

    @testset "splu(method=:multifrontal) solve & UMFPACK agreement" begin
        for A in (poisson2d(30), poisson3d(8), poisson3d(12), randmat(800, 6))
            n = size(A, 1)
            b = randn(n)
            x = splu(A; method = :multifrontal) \ b
            @test norm(A * x - b) / norm(b) <= 1e-8
            @test norm(x - lu(A) \ b) / norm(lu(A) \ b) <= 1e-7
        end
    end

    @testset "scalings & ComplexF64 through multifrontal" begin
        A = poisson3d(8)
        b = randn(size(A, 1))
        for sc in (SCALE_NONE, SCALE_SUM, SCALE_MAX)
            x = solve(splu(A; method = :multifrontal, scale = sc), b)
            @test norm(A * x - b) / norm(b) <= 1e-8
        end
        Ac = sprand(ComplexF64, 300, 300, 8 / 300) + (5 + 0im) * I
        Ac = Ac + Ac'                       # Hermitian-pattern, structurally symmetric
        bc = randn(ComplexF64, 300)
        xc = splu(Ac; method = :multifrontal) \ bc
        @test norm(Ac * xc - bc) / norm(bc) <= 1e-8
    end

    @testset "edge cases" begin
        @test (splu(sparse(reshape([4.0], 1, 1)); method = :multifrontal) \ [8.0])[1] ≈ 2.0
        D = sparse(Diagonal([2.0, 3.0, 5.0, 7.0]))
        @test splu(D; method = :multifrontal) \ [2.0, 6.0, 15.0, 28.0] ≈
              [1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError splu(D; method = :nonsense)
    end

    @testset "threaded == serial (identical fill & solution)" begin
        # Elimination-tree multithreading must produce a bit-identical factor to the
        # serial path: same structure, same numeric values, same solution.  Forcing
        # parallel_threshold = 0 engages the threaded driver regardless of size (a
        # no-op when run with a single thread, which still exercises the threaded
        # code path).  Comparing the two `multifrontal_lu` factorizations of the same
        # (unscaled) matrix isolates the threading from scaling.
        tsolve(F,
            b) = (n = length(b);
            c = collect(float.(b[F.p]));
            PureUMFPACK.lsolve!(F.L, c);
            PureUMFPACK.usolve!(F.U, c);
            x = similar(c);
            x[F.q] = c;
            x)
        for A in (poisson2d(24), poisson3d(8), poisson3d(12),
            randmat(800, 6), arrowband(800, 4))
            n = size(A, 1)
            Fs = multifrontal_lu(A; check = false, threaded = false)
            Ft = multifrontal_lu(A; check = false, threaded = true,
                parallel_threshold = 0)
            @test Fs.L.colptr == Ft.L.colptr && Fs.L.rowval == Ft.L.rowval
            @test Fs.U.colptr == Ft.U.colptr && Fs.U.rowval == Ft.U.rowval
            @test Fs.L.nzval == Ft.L.nzval && Fs.U.nzval == Ft.U.nzval
            @test Fs.p == Ft.p && Fs.q == Ft.q
            @test nnz(Fs.L) + nnz(Fs.U) == nnz(Ft.L) + nnz(Ft.U)
            b = randn(n)
            xs = tsolve(Fs, b)
            xt = tsolve(Ft, b)
            @test xs == xt
            @test norm(A * xt - b) / norm(b) <= 1e-8
        end
    end
end

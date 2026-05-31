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

    # Cross-front (delayed) threshold pivoting: genuinely unsymmetric matrices
    # with small/zero diagonal entries where the in-block path's restricted
    # pivoting is inadequate (it produces a NaN factor / SingularException) but
    # cross-front delay matches UMFPACK and the already-robust GP-LU path.
    @testset "delayed pivoting (cross-front)" begin
        # zero-diagonal random unsymmetric (in-block pivoting fails)
        zerodiag(n, p, seed) = begin
            rng = MersenneTwister(seed)
            A = sprandn(rng, n, n, p)
            for i in 1:n
                A[i, i] = 0.0
            end
            while abs(det(Matrix(A))) < 1e-2
                A = A + sprandn(rng, n, n, 0.1)
                for i in 1:n
                    A[i, i] = 0.0
                end
            end
            A
        end
        # row-permuted upper triangular (diagonal scrambled)
        function rowperm_upper(n, seed)
            rng = MersenneTwister(seed)
            U0 = triu(sprandn(rng, n, n, 0.6)) + n * I
            sparse(Matrix(U0)[randperm(rng, n), :])
        end

        unsym = [zerodiag(10, 0.5, 11), zerodiag(20, 0.35, 5),
            rowperm_upper(8, 3), rowperm_upper(15, 9)]

        @testset "in-block path is inadequate here" begin
            # At least one of these must defeat the restricted in-block pivoting
            # (NaN reconstruction or SingularException) — that is what motivates
            # the delayed path.
            failed = false
            for A in unsym
                bad = try
                    Fi = multifrontal_lu(A; delayed = false, check = false)
                    R = Matrix(A[Fi.p, Fi.q]) - Matrix(Fi.L * Fi.U)
                    !isfinite(norm(R)) || norm(R) > 1e-6 * max(1.0, norm(Matrix(A)))
                catch
                    true
                end
                failed |= bad
            end
            @test failed
        end

        @testset "delayed reconstruction A[p,q] = L*U" begin
            for A in unsym
                n = size(A, 1)
                F = multifrontal_lu(A; tol = 0.1, delayed = true, check = true)
                @test istril(F.L) && all(==(1.0), diag(F.L))
                @test istriu(F.U)
                @test isperm1(F.p, n) && isperm1(F.q, n)
                R = Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)
                @test norm(R) <= 1e-10 * max(1.0, norm(Matrix(A)))
            end
        end

        @testset "delayed matches UMFPACK and GP-LU (real residuals)" begin
            for A in unsym
                n = size(A, 1)
                b = randn(n)
                xd = splu(A; method = :multifrontal, delayed = true,
                    tol = 0.1, scale = SCALE_NONE) \ b
                xg = splu(A; method = :gplu, tol = 0.1, scale = SCALE_NONE) \ b
                xu = lu(A) \ b
                # real backward error of the delayed solve
                @test norm(A * xd - b) / norm(b) <= 1e-8
                # delayed agrees with both robust references
                @test norm(xd - xu) / norm(xu) <= 1e-7
                @test norm(xd - xg) / norm(xg) <= 1e-7
            end
        end

        @testset "SPD / diagonally-dominant still correct under delayed" begin
            for A in (poisson2d(12), poisson3d(6), randmat(200, 6))
                n = size(A, 1)
                b = randn(n)
                F = multifrontal_lu(A; tol = 0.1, delayed = true, check = true)
                R = Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)
                @test norm(R) <= 1e-9 * max(1.0, norm(Matrix(A)))
                x = splu(A; method = :multifrontal, delayed = true, tol = 0.1) \ b
                @test norm(A * x - b) / norm(b) <= 1e-8
            end
        end

        @testset "default (delayed=false) unchanged" begin
            # The default path must still match the in-block factorization exactly.
            for A in (poisson2d(8), randmat(300, 6))
                F0 = multifrontal_lu(A; delayed = false, check = false)
                F1 = multifrontal_lu(A; check = false)   # default
                @test F0.p == F1.p && F0.q == F1.q
                @test F0.L == F1.L && F0.U == F1.U
            end
        end
    end
end

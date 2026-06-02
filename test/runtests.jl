using PureUMFPACK
using SparseArrays, LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "..", "bench", "matrices.jl"))

Random.seed!(1234)

isperm1(p, n) = sort(p) == collect(1:n)

@testset "PureUMFPACK" begin
    @testset "gplu reconstruction  A[p,q] = L*U" begin
        for n in (1, 2, 10, 64, 300)
            for trial in 1:2
                A = sparse(sprand(n, n, min(1.0, 8 / n)) + (n) * I)
                F = gplu(A; tol = 0.1)
                @test istril(F.L) && all(==(1.0), diag(F.L))
                @test istriu(F.U)
                @test isperm1(F.p, n) && isperm1(F.q, n)
                R = Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)
                @test norm(R) <= 1.0e-9 * max(1.0, norm(Matrix(A)))
            end
        end
    end

    @testset "gplu strict partial pivoting on non-diagonally-dominant" begin
        for n in (50, 200)
            A = sparse(sprand(n, n, 10 / n) + 1.0 * I)
            b = randn(n)
            F = gplu(A; tol = 1.0)
            x = solve(F, b)
            @test norm(A * x - b) / norm(b) <= 1.0e-8
        end
    end

    @testset "sort_factors=false still solves" begin
        A = sparse(sprand(120, 120, 10 / 120) + 5.0 * I)
        b = randn(120)
        Fs = gplu(A; tol = 0.1, sort_factors = true)
        Fu = gplu(A; tol = 0.1, sort_factors = false)
        @test norm(solve(Fs, b) - solve(Fu, b)) <= 1.0e-10 * norm(solve(Fs, b))
    end

    @testset "sorted factors are canonical CSC (== double-transpose)" begin
        # row indices within each column must ascend, and the sorted factor must
        # equal the canonical form produced by a transpose-transpose round trip.
        colsorted(S) = all(
            issorted(@view rowvals(S)[SparseArrays.getcolptr(S)[j]:(SparseArrays.getcolptr(S)[j + 1] - 1)])
                for j in 1:size(S, 2)
        )
        canon(S) = copy(transpose(copy(transpose(S))))
        for A in (poisson3d(8), randmat(400, 10), arrowband(400, 4))
            F = gplu(A; q = amd_order_sym(A), tol = 0.1)
            @test colsorted(F.L) && colsorted(F.U)
            @test F.L == canon(F.L)
            @test F.U == canon(F.U)
        end
    end

    @testset "AMD / COLAMD validity and fill reduction" begin
        for A in (poisson2d(20), poisson3d(8), randmat(500, 6), arrowband(500, 3))
            n = size(A, 1)
            pa = amd_order_sym(A)
            pc = colamd_order(A)
            @test isperm1(pa, n)
            @test isperm1(pc, n)
        end
        # AMD must beat natural order on a Laplacian
        A = poisson3d(10)
        n = size(A, 1)
        M = A + A'
        fnat = (F = gplu(M; q = 1:n, tol = 0.0); nnz(F.L) + nnz(F.U))
        famd = (
            p = amd_order_sym(A);
            F = gplu(M[p, p]; q = 1:n, tol = 0.0);
            nnz(F.L) +
                nnz(F.U)
        )
        @test famd < fnat / 2
    end

    @testset "splu end-to-end solve (orderings × scalings)" begin
        for A in (poisson2d(30), poisson3d(10), randmat(800, 8), arrowband(800, 4))
            n = size(A, 1)
            b = randn(n)
            xref = Matrix(A) \ b
            for ord in (:natural, :amd, :colamd), sc in (SCALE_NONE, SCALE_SUM, SCALE_MAX)
                F = splu(A; ordering = ord, tol = 0.1, scale = sc)
                x = solve(F, b)
                @test norm(A * x - b) / norm(b) <= 1.0e-8
                @test norm(x - xref) / norm(xref) <= 1.0e-6
            end
        end
    end

    @testset "matches UMFPACK solution" begin
        for A in (poisson2d(25), randmat(600, 8))
            n = size(A, 1)
            b = randn(n)
            xu = lu(A) \ b
            xp = splu(A; ordering = :amd) \ b
            @test norm(xp - xu) / norm(xu) <= 1.0e-7
        end
    end

    @testset "iterative refinement does not degrade accuracy" begin
        A = sparse(sprand(400, 400, 12 / 400) + 1.0 * I)
        b = randn(400)
        F = splu(A; ordering = :amd, tol = 0.1)
        r0 = norm(A * solve(F, b) - b)
        r2 = norm(A * solve(F, b; refine = 2) - b)
        @test r2 <= r0 + 1.0e-12
    end

    @testset "edge cases" begin
        A1 = sparse(reshape([3.0], 1, 1))
        @test (splu(A1) \ [6.0])[1] ≈ 2.0
        D = sparse(Diagonal([2.0, 3.0, 5.0, 7.0]))
        b = [2.0, 6.0, 15.0, 28.0]
        @test splu(D) \ b ≈ [1.0, 2.0, 3.0, 4.0]
        @test isperm1(amd_order_sym(D), 4)
        @test isperm1(colamd_order(D), 4)
    end
end

# type-generality suite (ComplexF64, Float32, Int32 indices, singularity)
include(joinpath(@__DIR__, "robustness.jl"))

# supernodal multifrontal kernel
include(joinpath(@__DIR__, "multifrontal_test.jl"))

# package quality assurance (Aqua)
include(joinpath(@__DIR__, "qa.jl"))

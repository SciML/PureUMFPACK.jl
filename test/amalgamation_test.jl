# Relaxed supernode amalgamation (`relax` keyword of the multifrontal path):
# fundamental partition at relax = 0, per-column fill budget, exact symbolic/numeric
# fill agreement, reconstruction and solve at relax > 0.
using PureUMFPACK
using PureUMFPACK: multifrontal_lu, symbolic_mf, predicted_fill
using SparseArrays, LinearAlgebra, Random, Test
include(joinpath(@__DIR__, "..", "benchmark", "matrices.jl"))
Random.seed!(7)

nsuper(S) = length(S.sstart) - 1
supers(S) = ((S.sstart[k], S.sstart[k + 1] - 1) for k in 1:nsuper(S))
nzcount(idx, n) = (c = zeros(Int, n); foreach(i -> c[i] += 1, idx); c)
tridiag(n) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))

const CLASSES = (
    ("poisson2d(30)", poisson2d(30)),
    ("poisson3d(10)", poisson3d(10)),
    ("randmat(600,6)", randmat(600, 6)),
    ("arrowband(100,5)", arrowband(100, 5)),
    ("arrowband(400,4)", arrowband(400, 4)),
)

@testset "relaxed supernode amalgamation" begin
    @testset "relax = 0 is the fundamental partition" begin
        for (_, A) in CLASSES
            S = symbolic_mf(A; relax = 0)
            n = size(A, 1)
            cc = length.(S.colstruct)
            @test all(
                S.parent[c] == c + 1 && S.colstruct[c] == vcat(c + 1, S.colstruct[c + 1])
                    for (c1, c2) in supers(S) for c in c1:(c2 - 1)
            )
            nchild = zeros(Int, n)
            foreach(p -> p != 0 && (nchild[p] += 1), S.parent)
            starts = S.sstart[1:(end - 1)]
            @test !any(
                j -> j > 1 && S.parent[j - 1] == j && nchild[j] == 1 &&
                    cc[j - 1] == cc[j] + 1, starts
            )
        end
    end

    @testset "$name: relax = $r" for (name, A) in CLASSES, r in (1, 2, 4, 8)
        n = size(A, 1)
        S0 = symbolic_mf(A)
        S = symbolic_mf(A; relax = r)
        @test S.qf == S0.qf && S.parent == S0.parent
        @test issubset(S.sstart, S0.sstart)
        @test nsuper(S) <= nsuper(S0)
        extra = length.(S.colstruct) .- length.(S0.colstruct)
        @test all(c -> issubset(S0.colstruct[c], S.colstruct[c]), 1:n)
        @test maximum(extra) <= r
        @test all(
            S.colstruct[c] == vcat((c + 1):c2, S.colstruct[c2])
                for (c1, c2) in supers(S) for c in c1:(c2 - 1)
        )

        F0 = multifrontal_lu(A; check = false)
        F = multifrontal_lu(A; relax = r, check = false)
        @test predicted_fill(S) == nnz(F.L) + nnz(F.U)
        @test nzcount(findnz(F.L)[2], n) .- nzcount(findnz(F0.L)[2], n) == extra
        @test nzcount(rowvals(F.U), n) .- nzcount(rowvals(F0.U), n) == extra
        @test sum(extra) <= r * n
        @test norm(Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)) <=
            1.0e-9 * max(1.0, norm(Matrix(A)))

        b = randn(n)
        x = splu(A; method = :multifrontal, relax = r) \ b
        @test norm(A * x - b) / norm(b) <= 1.0e-8
        @test norm(x - lu(A) \ b) / norm(lu(A) \ b) <= 1.0e-7
    end

    @testset "amalgamation merges supernodes on Poisson problems" begin
        for A in (poisson2d(30), poisson3d(10))
            @test nsuper(symbolic_mf(A; relax = 4)) < nsuper(symbolic_mf(A))
        end
    end

    @testset "long chain stays within the per-column budget" begin
        n = 200
        A = tridiag(n)
        S0 = symbolic_mf(A; q = 1:n)
        @test nsuper(S0) == n - 1
        for r in (1, 3)
            S = symbolic_mf(A; q = 1:n, relax = r)
            @test maximum(length.(S.colstruct) .- length.(S0.colstruct)) <= r
            @test maximum(c2 - c1 + 1 for (c1, c2) in supers(S)) <= r + 2
            F = multifrontal_lu(A; q = 1:n, relax = r)
            F0 = multifrontal_lu(A; q = 1:n)
            @test nnz(F.L) - nnz(F0.L) <= r * n
            @test norm(Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)) <= 1.0e-12 * norm(Matrix(A))
        end
    end

    @testset "negative relax is rejected" begin
        A = poisson2d(4)
        @test_throws ArgumentError symbolic_mf(A; relax = -1)
        @test_throws ArgumentError multifrontal_lu(A; relax = -1)
        @test_throws ArgumentError splu(A; method = :multifrontal, relax = -1)
    end
end

# Type-generality robustness: a faithful UMFPACK translation must handle
# ComplexF64, Float32, and Int32 indices, and detect structural singularity.
using PureUMFPACK
using SparseArrays, LinearAlgebra, Random, Test
Random.seed!(7)

@testset "type generality" begin
    @testset "ComplexF64" begin
        n = 400
        A = sprand(ComplexF64, n, n, 8 / n) + (5 + 0im) * I
        b = randn(ComplexF64, n)
        for ord in (:amd, :colamd, :natural)
            x = splu(A; ordering = ord) \ b
            @test norm(A * x - b) / norm(b) <= 1.0e-10
        end
        @test norm((splu(A) \ b) - (lu(A) \ b)) / norm(lu(A) \ b) <= 1.0e-9
    end
    @testset "Float32" begin
        n = 300
        A = sparse(sprand(Float32, n, n, 8 / n) + 5.0f0 * I)
        b = randn(Float32, n)
        F = splu(A; ordering = :amd)
        x = F \ b
        @test eltype(x) == Float32
        @test norm(A * x - b) / norm(b) <= 1.0f-4
    end
    @testset "Int32 indices" begin
        n = 300
        A0 = sparse(sprand(n, n, 8 / n) + 6I)
        A = SparseMatrixCSC{Float64, Int32}(A0)
        b = randn(n)
        F = splu(A; ordering = :amd)
        @test eltype(F.q) == Int32
        @test norm(A0 * (F \ b) - b) / norm(b) <= 1.0e-9
        Fc = splu(A; ordering = :colamd)
        @test norm(A0 * (Fc \ b) - b) / norm(b) <= 1.0e-9
    end
    @testset "structural singularity is detected" begin
        A = sparse([1.0 2.0; 2.0 4.0])          # rank 1
        @test_throws SingularException gplu(A; tol = 1.0, check = true)
        B = spdiagm(0 => [1.0, 0.0, 1.0])        # zero pivot
        @test_throws SingularException splu(B; ordering = :natural, scale = SCALE_NONE)
    end
end

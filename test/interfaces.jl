using PureUMFPACK
using SparseArrays, Test

@testset "factorization generic interfaces" begin
    A = sparse([2.0 1.0; 1.0 2.0])
    b = [1.0, 0.0]

    for F in (gplu(A), splu(A), multifrontal_lu(A))
        @test size(F) == size(A)
        @test size(F, 1) == size(A, 1)
        @test size(F, 2) == size(A, 2)
        @test F \ b ≈ A \ b
    end
end

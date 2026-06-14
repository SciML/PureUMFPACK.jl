using PureUMFPACK, Aqua, Test

@testset "Aqua quality assurance" begin
    Aqua.test_all(PureUMFPACK)
end

using SafeTestsets

@safetestset "Aqua quality assurance" begin
    using PureUMFPACK, Aqua, Test
    Aqua.test_all(PureUMFPACK)
end

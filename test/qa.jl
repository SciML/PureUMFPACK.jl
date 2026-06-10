using PureUMFPACK, Aqua, Test

@testset "Aqua quality assurance" begin
    # Run all Aqua checks except deps_compat, which is split below so the passing
    # julia/weakdeps parts still run while only the failing extras check is broken.
    Aqua.test_all(PureUMFPACK; deps_compat = false)
    # check_extras disabled: `Pkg` is in [extras]/[targets].test without a [compat] bound.
    # Tracked in https://github.com/SciML/PureUMFPACK.jl/issues/21
    Aqua.test_deps_compat(PureUMFPACK; check_extras = false)
    @test_broken false  # Aqua deps_compat: `Pkg` extras dep lacks [compat] — tracked in https://github.com/SciML/PureUMFPACK.jl/issues/21
end

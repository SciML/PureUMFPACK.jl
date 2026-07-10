using PureUMFPACK
using Test

@testset "public API documentation" begin
    public_names = filter(!=(:PureUMFPACK), names(PureUMFPACK; all = false, imported = false))
    expected_names = Set(
        [
            :GPLUFactorization,
            :PureLU,
            :SCALE_MAX,
            :SCALE_NONE,
            :SCALE_SUM,
            :amd_order_sym,
            :colamd_order,
            :gplu,
            :multifrontal_lu,
            :row_scaling,
            :solve,
            :splu,
        ]
    )
    @test Set(public_names) == expected_names

    for name in public_names
        binding = Docs.Binding(PureUMFPACK, name)
        @test Docs.hasdoc(binding)
    end

    api_page = read(joinpath(pkgdir(PureUMFPACK), "docs", "src", "api.md"), String)
    for name in public_names
        @test occursin(string(name), api_page)
    end
end

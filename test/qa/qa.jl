using SciMLTesting, PureUMFPACK, Test

run_qa(
    PureUMFPACK;
    explicit_imports = true,
    ei_kwargs = (;
        # `Base.OneTo` (gplu.jl) is not declared public in Base.
        all_qualified_accesses_are_public = (; ignore = (:OneTo,)),
        # `SparseArrays.getcolptr` is the CSC colptr accessor, not yet declared public.
        all_explicit_imports_are_public = (; ignore = (:getcolptr,)),
    ),
)

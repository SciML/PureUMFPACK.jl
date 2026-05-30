using Documenter, PureUMFPACK

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md");
    force = true)

makedocs(;
    modules = [PureUMFPACK],
    authors = "Chris Rackauckas <accounts@chrisrackauckas.com> and contributors",
    sitename = "PureUMFPACK.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/PureUMFPACK/stable/",
        assets = String[]),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ],
    checkdocs = :exports)

deploydocs(;
    repo = "github.com/SciML/PureUMFPACK.jl",
    devbranch = "master",
    push_preview = true)

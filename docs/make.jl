using AffineRayleighOptimization
using Documenter

DocMeta.setdocmeta!(AffineRayleighOptimization, :DocTestSetup, :(using AffineRayleighOptimization); recursive=true)

makedocs(;
    modules=[AffineRayleighOptimization],
    authors="William Samuelson and Viktor Svensson", sitename="AffineRayleighOptimization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://williamesamuelson.github.io/AffineRayleighOptimization.jl",
        edit_link="main",
        assets=String[],
        repolink="https://github.com/williamesamuelson/AffineRayleighOptimization.jl/blob/{commit}{path}#{line}",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/williamesamuelson/AffineRayleighOptimization.jl",
    devbranch="main",
)

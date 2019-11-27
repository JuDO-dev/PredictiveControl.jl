using Documenter, PredictiveControl

makedocs(;
    modules=[PredictiveControl],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/imciner2/PredictiveControl.jl/blob/{commit}{path}#L{line}",
    sitename="PredictiveControl.jl",
    authors="Ian McInerney, Imperial College London",
    assets=String[],
)

deploydocs(;
    repo="github.com/imciner2/PredictiveControl.jl",
)

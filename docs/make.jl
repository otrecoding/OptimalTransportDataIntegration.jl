using Documenter
using OptimalTransportDataIntegration

makedocs(
    sitename = "OptimalTransportDataIntegration.jl",
    authors = "Jeremy Omer <jeremy.omer@insa-rennes.fr>, Valérie Garès <valerie.gares@inria.fr> and Pierre Navaro <navaro@math.cnrs.fr>",
    format = Documenter.HTML(; canonical = "https://otrecoding.github.io/OptimalTransportDataIntegration.jl"),
    modules = [OptimalTransportDataIntegration],
    pages = [
        "Home" => "index.md",
        "Transport within a data source" => "joint_ot_within_base.md",
        "Transport between data sources" => "joint_ot_between_bases.md",
        "Machine Learning" => "learning.md",
        "Numerical experiments" => "simulations.md",
    ],
    doctest = false,
    warnonly = Documenter.except(),
)

deploydocs(repo = "github.com/otrecoding/OptimalTransportDataIntegration.jl.git")

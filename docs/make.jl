using Documenter
using OptimalTransportDataIntegration

makedocs(
    sitename = "OptimalTransportDataIntegration",
    authors = "Jeremy Omer, Valérie Garès and Pierre Navaro",
    format = Documenter.HTML(),
    modules = [OptimalTransportDataIntegration],
    pages = [
        "Quickstart" => "index.md",
        "Transport within a data source" => "joint_ot_within_base.md",
        "Transport between data sources" => "joint_ot_between_bases.md",
        "Machine Learning" => "learning.md",
        "Numerical experiments" => "simulations.md",
    ],
    doctest = false,
    warnonly = Documenter.except(),
)

deploydocs(repo = "github.com/otrecoding/OptimalTransportDataIntegration.jl.git")

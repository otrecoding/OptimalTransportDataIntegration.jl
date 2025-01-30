using Documenter
using OptimalTransportDataIntegration

makedocs(
    sitename = "OptimalTransportDataIntegration",
    authors = "Jeremy Omer, Valérie Garès and Pierre Navaro",
    format = Documenter.HTML(),
    modules = [OptimalTransportDataIntegration],
    pages = ["Documentation" => "index.md"],
    doctest = false
)

deploydocs(repo = "github.com/otrecoding/OptimalTransportDataIntegration.jl.git")

using Documenter
using OptimalTransportDataIntegration

makedocs(
    sitename = "OptimalTransportDataIntegration",
    authors = "Jeremy Omer, Valérie Garès and Pierre Navaro",
    format = Documenter.HTML(),
    modules = [OTRecod],
    pages = [
        "Documentation" => "index.md"
    ],
)

deploydocs(
    repo = "github.com/otrecoding/OptimalTransportDataIntegration.jl.git",
)

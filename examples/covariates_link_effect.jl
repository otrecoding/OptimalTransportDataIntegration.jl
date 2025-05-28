# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,jl
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Julia 1.11.1
#     language: julia
#     name: julia-1.11
# ---

using DelimitedFiles
using OptimalTransportDataIntegration

# +
function covariates_link_effect(nsimulations::Int, pvalues)

    outfile = "covariates_link_effect.csv"
    header = ["id", "p", "estyb", "estza", "estimation", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for p in pvalues

            params = DataParameters(p = p)

            rng = DataGenerator(params)

            for i = 1:nsimulations

                data = generate(rng)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i p estyb estza est "ot"])

                #OT-r Regularized Transport 
                result = otrecod(data, JointOTWithinBase())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i p estyb estza est "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg_m1 = 0.0, reg_m2 = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i p estyb estza est "ote"])

                #OTE Regularized unbalanced transport 
                result = otrecod(data, JointOTBetweenBases())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i p estyb estza est "ote-r"])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i p estyb estza est "sl"])

            end

        end

    end

end

nsimulations = 1000

@time covariates_link_effect(nsimulations, (0.2, 0.4, 0.6, 0.8))

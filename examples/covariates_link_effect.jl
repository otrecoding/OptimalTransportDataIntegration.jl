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
function covariates_link_effect(nsimulations::Int, r2values)

    outfile = "covariates_link_effect.csv"
    header = ["id", "r2", "estyb", "estza", "accuracy", "method",  "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for r2 in r2values, scenario in (1, 2)

            params = DataParameters(r2 = r2)

            rng = DataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "ot" scenario])

                #OT-r Regularized Transport
                result = otrecod(data, JointOTWithinBase())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "ot-r" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg_m1 = 0.0, reg_m2 = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "ote" scenario])

                #OTE Regularized unbalanced transport
                result = otrecod(data, JointOTBetweenBases())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "ote-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "sl" scenario])

            end

        end

    end

end

nsimulations = 100

@time covariates_link_effect(nsimulations, (0.2, 0.4, 0.6, 0.8))

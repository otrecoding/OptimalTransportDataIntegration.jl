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
function sample_size_effect(all_params, nsimulations)

    outfile = "sample_size_effect.csv"
    header = ["id", "nA", "nB", "estyb", "estza", "estimation", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for params in all_params, scenario in (1, 2)

            rng = DataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)

                writedlm(io, [i params.nA params.nB estyb estza est "wi" scenario])


                #OT-r Regularized Transport
                result = otrecod(data, JointOTWithinBase())
                estyb, estza, est = accuracy(result)

                writedlm(io, [i params.nA params.nB estyb estza est "wi-r" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg_m1 = 0.0, reg_m2 = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be-un" scenario])

                #OTE Regularized unbalanced transport
                result = otrecod(data, JointOTBetweenBases())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be-un-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)

                writedlm(io, [i params.nA params.nB estyb estza est "sl"  scenario])


            end

        end

    end

end

all_params = [
    DataParameters(nA = 100, nB = 100),
    DataParameters(nA = 1000, nB = 1000),
    DataParameters(nA = 10000, nB = 10000),
]

nsimulations = 100

@time sample_size_effect(all_params, nsimulations)

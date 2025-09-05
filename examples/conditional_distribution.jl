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
function conditional_distribution(nsimulations::Int, epsilons)

    outfile = "conditional_distribution.csv"
    header = ["id", "epsilon", "estyb", "estza", "accuracy", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        params = DataParameters(mB = [0, 0, 0])

        for scenario in (1, 2)

            rng = DiscreteDataGenerator(params, scenario = scenario)

            for eps in epsilons

                for i in 1:nsimulations

                    data = generate(rng, eps = eps)

                    #OT Transport of the joint distribution of covariates and outcomes.
                    alpha, lambda = 0.0, 0.0
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i eps estyb estza est "wi" scenario])

                    #OT-r Regularized Transport
                    result = otrecod(data, JointOTWithinBase())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i eps estyb estza est "wi-r" scenario])

                    #OTE Balanced transport of covariates and estimated outcomes
                    result = otrecod(data, JointOTBetweenBases(reg_m1 = 0.0, reg_m2 = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i eps estyb estza est "be-un" scenario])

                    #OTE Regularized unbalanced transport
                    result = otrecod(data, JointOTBetweenBases())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i eps estyb estza est "be-un-r" scenario])

                    #SL Simple Learning
                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i eps estyb estza est "sl" scenario])

                end

            end

        end

    end

end

nsimulations = 100
epsilons = (0.0, 0.1, 0.5, 1.0)

@time conditional_distribution(nsimulations, epsilons)

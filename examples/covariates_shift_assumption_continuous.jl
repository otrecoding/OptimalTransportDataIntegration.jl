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
function covariates_shift_assumption_continuous(nsimulations::Int, shift)

    outfile = "covariates_shift_assumption_continuous.csv"
    header = ["id", "mB", "estyb", "estza", "estimation", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in shift

            params = DataParameters(mB = mB)

            for scenario in 1:2

                rng = ContinuousDataGenerator(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)

                    #OT Transport of the joint distribution of covariates and outcomes.
                    alpha, lambda = 0.1, 0.1
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "within" scenario])

                    #OTE Regularized unbalanced transport
                    result = otrecod(data, JointOTBetweenBases())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "between" scenario])

                    #SL Simple Learning
                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "sl" scenario])

                end

            end

        end

    end

end

nsimulations = 100
shift = ([5, 5, 5], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0])

@time covariates_shift_assumption_continuous(nsimulations, shift)

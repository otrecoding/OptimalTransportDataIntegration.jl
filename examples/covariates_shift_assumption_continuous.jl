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
function covariates_shift_assumption_continuous(nsimulations::Int, scenarios)

    outfile = "covariates_shift_assumption_continuous.csv"
    header = ["id", "mB", "estyb", "estza", "estimation", "method"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in scenarios

            params = DataParameters(mB = mB)
            rng = DataGenerator(params, scenario = 1, discrete = false)

            for i in 1:nsimulations

                data = generate(rng)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.6, 0.2
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i repr(mB) estyb estza est "within"])

                #OTE Regularized unbalanced transport
                result = otrecod(data, JointOTBetweenBases())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i repr(mB) estyb estza est "between"])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i repr(mB) estyb estza est "sl"])

            end

        end

    end

end

nsimulations = 1000
scenarios = ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0])

@time covariates_shift_assumption_continuous(nsimulations, scenarios)

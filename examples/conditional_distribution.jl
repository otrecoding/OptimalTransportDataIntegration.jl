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
function conditional_distribution(nsimulations::Int, scenarios)

    outfile = "conditional_distribution.csv"
    header = ["id", "epsilon", "estyb", "estza", "accuracy", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        params = DataParameters(mB = [0, 0, 0])

        rng = PDataGenerator( params, scenario = 1 )

        for eps in scenarios

            for i = 1:nsimulations

                data = generate_data(rng, eps = eps)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy( result )
                writedlm(io, [i eps estyb estza est "ot"])

                #OT-r Regularized Transport 
                alpha, lambda = 0.4, 0.1
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy( result )
                writedlm(io, [i eps estyb estza est "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg = 0.001, reg_m1 = 0.0, reg_m2 = 0.0))
                estyb, estza, est = accuracy( result )
                writedlm(io, [i eps estyb estza est "ote"])

                #OTE Regularized unbalanced transport 
                result = otrecod(data, JointOTBetweenBases(reg = 0.001, reg_m1 = 0.01, reg_m2 = 0.01))
                estyb, estza, est = accuracy( result )
                writedlm(io, [i eps estyb estza est "ote-r"])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy( result )
                writedlm(io, [i eps estyb estza est "sl"])

            end

        end

    end

end

nsimulations = 1000
scenarios = (0.0, 0.1, 0.5, 1.0)

@time conditional_distribution(nsimulations, scenarios)

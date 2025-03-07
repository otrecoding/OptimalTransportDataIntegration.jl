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
    header = ["id", "epsilon", "estimation", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for eps in scenarios

            params = DataParameters(eps = eps, mB = [0, 0, 0])

            for i = 1:nsimulations

                data = generate_xcat_ycat(params)

                #OT Transport of the joint distribution of covariates and outcomes.
                maxrelax, lambda_reg = 0.0, 0.0
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i eps est "ot"])

                #OT-r Regularized Transport 
                maxrelax, lambda_reg = 0.4, 0.1
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i eps est "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                est = otrecod(data, UnbalancedModality(reg = 0.0, reg_m1 = 0.0, reg_m2 = 0.0))
                writedlm(io, [i eps est "ote"])

                #OTE Regularized unbalanced transport 
                est = otrecod(data, UnbalancedModality(reg = 0.0, reg_m1 = 0.01, reg_m2 = 0.01))
                writedlm(io, [i eps est "ote-r"])

                #SL Simple Learning
                est = otrecod(data, SimpleLearning())
                writedlm(io, [i eps est "sl"])

            end

        end

    end

end

nsimulations = 1000
scenarios = (0.0, 0.1, 0.5, 1.0)

@time conditional_distribution(nsimulations, scenarios)

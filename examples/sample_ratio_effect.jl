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
function sample_ratio_effect(nsimulations::Int, ratios)

    outfile = "sample_ratio_effect.csv"
    header = ["id", "nA", "nB", "estimation", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for r in ratios

            nA = 1000
            nB = nA ÷ r
            params = DataParameters(nB = nB)

            for i = 1:nsimulations

                data = generate_xcat_ycat(params)

                #OT Transport of the joint distribution of covariates and outcomes.
                maxrelax, lambda_reg = 0.0, 0.0
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i params.nA params.nB est "ot"])

                #OT-r Regularized Transport 
                maxrelax, lambda_reg = 0.4, 0.1
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i params.nA params.nB est "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                est = otrecod(data, UnbalancedModality(reg = 0.01, reg_m = 0.0))
                writedlm(io, [i params.nA params.nB est "ote"])

                #OTE Regularized unbalanced transport 
                est = otrecod(data, UnbalancedModality(reg = 0.01, reg_m = 0.05))
                writedlm(io, [i params.nA params.nB est "ote-r"])

                #SL Simple Learning
                est = otrecod(data, SimpleLearning())
                writedlm(io, [i params.nA params.nB est "sl"])

            end

        end

    end

end

nsimulations = 100

@time sample_ratio_effect(nsimulations, (1, 2, 10))

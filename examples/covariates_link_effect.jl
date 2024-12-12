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
    header = ["id", "p", "estimation", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for p in pvalues

            params = DataParameters(p = p)

            for i = 1:nsimulations

                data = generate_xcat_ycat(params)

                #OT Transport of the joint distribution of covariates and outcomes.
                maxrelax, lambda_reg = 0.0, 0.0
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i p est "ot"])

                #OT-r Regularized Transport 
                maxrelax, lambda_reg = 0.4, 0.1
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i p est "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                est = otrecod(data, UnbalancedModality(reg = 0.01, reg_m = 0.0))
                writedlm(io, [i p est "ote"])

                #OTE Regularized unbalanced transport 
                est = otrecod(data, UnbalancedModality(reg = 0.01, reg_m = 0.05))
                writedlm(io, [i p est "ote-r"])

                #SL Simple Learning
                est = otrecod(data, SimpleLearning())
                writedlm(io, [i p est "sl"])

            end

        end

    end

end

nsimulations = 100

@time covariates_link_effect(nsimulations, (0.2, 0.4, 0.6, 0.8))

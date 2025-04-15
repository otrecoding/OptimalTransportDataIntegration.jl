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
    header = ["id", "nA", "nB", "estyb", "estza", "estimation", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for params in all_params

            for i = 1:nsimulations

                data = generate_xcat_ycat(params)

                #OT Transport of the joint distribution of covariates and outcomes.
                maxrelax, lambda_reg = 0.0, 0.0
                yb, za = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza "ot"])

                #OT-r Regularized Transport 
                maxrelax, lambda_reg = 0.4, 0.1
                yb, za = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                yb, za = otrecod(data, UnbalancedModality(reg = 0.0, reg_m1 = 0.0, reg_m2 = 0.0))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza "ote"])

                #OTE Regularized unbalanced transport 
                yb, za = otrecod(data, UnbalancedModality(reg = 0.0, reg_m1 = 0.01, reg_m2 = 0.01))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza "ote-r"])

                #SL Simple Learning
                yb, za = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza "sl"])

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

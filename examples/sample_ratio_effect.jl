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
    header = ["id", "nA", "nB", "estya", "estzb", "estimation", "method"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for r in ratios

            nA = 1000
            nB = nA ÷ r
            params = DataParameters(nB = nB)

            for i = 1:nsimulations

                data = generate_data(params)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                yb, za = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza est "ot"])

                #OT-r Regularized Transport 
                alpha, lambda = 0.4, 0.1
                yb, za = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza est "ot-r"])

                #OTE Balanced transport of covariates and estimated outcomes
                yb, za = otrecod(data, JointOTBetweenBases(reg = 0.0, reg_m1 = 0.0, reg_m2 = 0.0))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza est "ote"])

                #OTE Regularized unbalanced transport 
                yb, za = otrecod(data, JointOTBetweenBases(reg = 0.0, reg_m1 = 0.01, reg_m2 = 0.01))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza est "ote-r"])

                #SL Simple Learning
                yb, za = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i params.nA params.nB estyb estza est "sl"])

            end

        end

    end

end

nsimulations = 1000

@time sample_ratio_effect(nsimulations, (1, 2, 5, 10))

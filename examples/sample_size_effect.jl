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

    estimations = Float32[]
    
    outfile =  "sample_size_effect_ot.csv"
    header = ["id", "nA", "nB", "maxrelax", "lambda_reg", "estimation"]

    open(outfile, "a") do io

        seekstart(io)
        writedlm(io, hcat(header...))
        for params in all_params

            for i in 1:nsimulations
            
                data = generate_xcat_ycat(params)
                maxrelax,  lambda_reg = 0.0, 0.0
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i params.nA params.nB  maxrelax lambda_reg est ])
                maxrelax,  lambda_reg = 0.4, 0.1
                est = otrecod(data, OTjoint(maxrelax = maxrelax, lambda_reg = lambda_reg))
                writedlm(io, [i params.nA params.nB  maxrelax lambda_reg est ])
            
            end

        end

    end

end

all_params = (
    DataParameters(nA = 100, nB = 100),
    DataParameters(nA = 1000, nB = 1000),
    DataParameters(nA = 10000, nB = 10000)
)

nsimulations = 100

@time sample_size_effect(nsimulations, all_params)

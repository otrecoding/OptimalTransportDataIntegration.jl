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
# ---

using OptimalTransportDataIntegration
using DataFrames
using CSV
using Printf
using DelimitedFiles

function otjoint(nsimulations)

    maxrelax = collect(0:0.1:2)
    lambda_reg = collect(0:0.1:1)
    estimations = Float32[]
    
    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], eps = 0.0, p = 0.2)
    
    outfile =  open("results_otjoint.csv","w")
    header = header = ["id", "maxrelax", "lambda_reg", "estimation"]
    writedlm("results.csv", hcat(header...))
    open("results.csv", "a") do io
        for i in 1:nsimulations
        
            data = generate_xcat_ycat(params)
            csv_file = @sprintf "dataset%04i.csv" i

            CSV.write(joinpath("datasets", csv_file), data)

            @show csv_file
            for m in maxrelax, λ in lambda_reg
        
                est = otrecod(data, OTjoint(maxrelax = m, lambda_reg = λ))
        
                writedlm(io, [i m λ est "otjoint"])
        
            end

            # est = otrecod(data, UnbalancedModality(reg = 0.0, reg_m = 0.0))

            #reg = [0.0, 0.001, 0.01, 0.1]
            #reg_m = [0.0 0.01 0.05 0.1 0.25 0.5 0.75 1]

            #for r in reg, r_m in reg_m

            #    est = otrecod(data, UnbalancedModality(reg = r, reg_m = r_m))

            #end

            #est = otrecod(data, SimpleLearning()) > 0.5

        end
    end

end

otjoint(1000)

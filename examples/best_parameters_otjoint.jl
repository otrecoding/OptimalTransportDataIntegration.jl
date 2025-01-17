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

function otjoint(start, stop)

    maxrelax = collect(0:0.1:2)
    lambda_reg = collect(0:0.1:1)
    estimations = Float32[]

    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], eps = 0.0, p = 0.2)

    outfile = "results_otjoint.csv"
    header = ["id", "maxrelax", "lambda_reg", "estimation", "method"]

    open(outfile, "a") do io


        for i = 1:nsimulations

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            data = generate_xcat_ycat(params)
            csv_file = @sprintf "dataset%04i.csv" i
            @show csv_file

            CSV.write(joinpath("datasets", csv_file), data)

            for m in maxrelax, λ in lambda_reg

                est = otrecod(data, OTjoint(maxrelax = m, lambda_reg = λ))
                writedlm(io, [i m λ est "otjoint"])

            end

        end

    end

end

otjoint(1, 1000)

##
data = CSV.read("results_otjoint.csv", DataFrame)

sort(
    combine(groupby(data, ["maxrelax", "lambda_reg"]), :estimation => mean),
    order(:estimation_mean, rev = true),
)

# equivalent with pandas
# import pandas as pd
# data = pd.read_csv("results_otjoint.csv", sep="\t")
# data.groupby(["maxrelax", "lambda_reg"]).estimation.mean().sort_values(ascending=False)

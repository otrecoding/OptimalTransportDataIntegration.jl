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
import Statistics: mean

function otjoint(start, stop)

    alpha = collect(0:0.1:2)
    lambda = collect(0:0.1:1)
    estimations = Float32[]

    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0])

    rng = DataGenerator(params)

    outfile = "results_otjoint.csv"
    header = ["id", "alpha", "lambda", "estimation", "method"]

    open(outfile, "a") do io


        for i = start:stop

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            data = generate_data(rng)
            csv_file = @sprintf "dataset%04i.csv" i
            @show csv_file

            CSV.write(joinpath("datasets", csv_file), data)

            for m in alpha, λ in lambda

                result = otrecod(data, JointOTWithinBase(alpha = m, lambda = λ))
                est_yb, est_za, est = accuracy(result)
                writedlm(io, [i m λ est "otjoint"])

            end

        end

    end

end

otjoint(1, 1000)

##
# data = CSV.read("results_otjoint.csv", DataFrame)

# sort(
#     combine(groupby(data, ["alpha", "lambda"]), :estimation => mean),
#     order(:estimation_mean, rev = true),
# )

# equivalent with pandas
# import pandas as pd
# data = pd.read_csv("results_otjoint.csv", sep="\t")
# data.groupby(["alpha", "lambda"]).estimation.mean().sort_values(ascending=False)

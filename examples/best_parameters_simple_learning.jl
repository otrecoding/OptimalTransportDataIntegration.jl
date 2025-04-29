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

function simple_learning(start, stop)

    estimations = Float32[]

    outfile = "results_simple_learning.csv"
    header = ["id" "estimation" "method"]

    open(outfile, "a") do io


        for i = start:stop

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            csv_file = @sprintf "dataset%04i.csv" i
            @show csv_file

            data = CSV.read(joinpath("datasets", csv_file), DataFrame)

            result = otrecod(data, SimpleLearning())
            est_yb, est_za, est = accuracy(result)

            writedlm(io, [i est "learning"])

        end

    end

end

simple_learning(1, 1000)

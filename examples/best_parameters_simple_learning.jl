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

function simple_learning(nsimulations::Int)

    estimations = Float32[]
    
    outfile =  "results_simple_learning.csv"
    header = ["id" "estimation" "method"]

    open(outfile, "a") do io

        seekstart(io)
        writedlm(io, hcat(header...))

        for i in 1:nsimulations
        
            csv_file = @sprintf "dataset%04i.csv" i
            @show csv_file

            data = CSV.read(joinpath("datasets", csv_file), DataFrame)

            @show est = otrecod(data, SimpleLearning())
            writedlm(io, [i est "learning"])

        end

    end

end

simple_learning(1000)

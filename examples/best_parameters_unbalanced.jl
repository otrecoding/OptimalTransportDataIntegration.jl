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

function unbalanced(start, stop)

    reg = [0.0, 0.001, 0.01, 0.1]
    reg_m = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    estimations = Float32[]

    outfile = "results_unbalanced.csv"
    header = ["id" "reg" "reg_m" "estimation" "method"]

    open(outfile, "a") do io

        for i = start:stop

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            csv_file = @sprintf "dataset%04i.csv" i
            @show csv_file

            data = CSV.read(joinpath("datasets", csv_file), DataFrame)

            for r in reg, r_m in reg_m

                @show est = otrecod(data, UnbalancedModality(reg = r, reg_m = r_m))
                writedlm(io, [i r r_m est "unbalanced"])

            end

        end

    end

end

unbalanced(968, 1000)

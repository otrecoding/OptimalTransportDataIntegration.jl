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

    reg = [0.001, 0.01, 0.1]
    reg_m1 = [0.0]
    reg_m2 = [0.0]
    estimations = Float32[]

    outfile = "results_balanced.csv"
    header = ["id" "reg" "estimation" "method"]

    open(outfile, "a") do io

        for i = start:stop

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            csv_file = @sprintf "dataset%04i.csv" i
            @show csv_file

            data = CSV.read(joinpath("datasets", csv_file), DataFrame)

            for r in reg, r_m1 in reg_m1, r_m2 in reg_m2

                yb, za = otrecod(data, JointOTBetweenBases(reg = r, reg_m1 = r_m1, reg_m2 = r_m2))
                estyb, estza, est = accuracy(data, yb, za)
                writedlm(io, [i r estyb estza est "balanced"])

            end

        end

    end

end

unbalanced(1, 1000)

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
    reg_m1 = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    reg_m2 = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    estimations = Float32[]

    outfile = "results_unbalanced.csv"
    header = ["id" "reg" "reg_m1" "reg_m2" "estimation" "method"]

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

                @show est = otrecod(data, UnbalancedModality(reg = r, reg_m1 = r_m1, reg_m2 = r_m2))
                writedlm(io, [i r r_m1 r_m2 est "unbalanced"])

            end

        end

    end

end

unbalanced(1, 1000)

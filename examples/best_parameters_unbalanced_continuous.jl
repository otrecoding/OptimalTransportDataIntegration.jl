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

function unbalanced_continuous(start, stop)

    reg = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    reg_m = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    outfile = "results_unbalanced_continuous.csv"
    header = ["id" "reg" "reg_m" "estyb" "estza" "est" "method"]
    params = DataParameters()
    rng = DataGenerator(params, scenario = 2, discrete = false)

    return open(outfile, "a") do io

        for i in start:stop

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            data = generate(rng)

            for r in reg, r_m in reg_m

                result = otrecod( data, JointOTBetweenBases(reg = r, reg_m1= r_m, reg_m2 = r_m))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r r_m estyb estza est "be-un-r"])

            end

        end

    end

end

unbalanced_continuous(1, 1000)

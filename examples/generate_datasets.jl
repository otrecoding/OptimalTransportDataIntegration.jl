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

import CSV
import JSON
using DataFrames
using OptimalTransportDataIntegration
using Printf

# +
function write_datasets(nsimulations, all_params, outdir)

    mkpath(outdir)

    for (j, params) in enumerate(all_params)

        rng = DataGenerator(params)

        for i = 1:nsimulations

            df = generate_data(rng)

            json_file = @sprintf "tab_otjoint_%02i_%02i.json" j i
            save_params(joinpath(outdir, json_file), params)

            csv_file = @sprintf "tab_otjoint_%02i_%02i.csv" j i

            CSV.write(joinpath(outdir, csv_file), df)

        end

    end

end
# -

all_params = [
    DataParameters(nA = 1000, nB = 1000, mB = [0, 0, 0]),
    DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0]),
    DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0]),
    DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0]),
    DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0]),
    DataParameters(nA = 1000, nB = 1000, mB = [1, 1, 0]),
    DataParameters(nA = 1000, nB = 1000, mB = [1, 2, 0]),
    DataParameters(nA = 100,  nB = 100,  mB = [1, 0, 0]),
    DataParameters(nA = 5000, nB = 5000, mB = [1, 0, 0]),
    DataParameters(nA = 1000, nB = 500,  mB = [1, 0, 0]),
    DataParameters(nA = 1000, nB = 100,  mB = [1, 0, 0]),
]

nsimulations = 10
outdir = joinpath("datasets")
write_datasets(nsimulations, all_params, outdir)

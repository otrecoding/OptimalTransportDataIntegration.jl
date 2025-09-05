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

params = DataParameters(nA = 1000, nB = 1000, mB = [0, 0, 0], p = 0.2)

rng = DiscreteDataGenerator(params)
data = generate(rng, eps = 0.01)

outdir = @__DIR__
json_file = "dataset.json"
save_params(joinpath(outdir, json_file), params)
csv_file = "dataset.csv"
CSV.write(joinpath(outdir, csv_file), data)

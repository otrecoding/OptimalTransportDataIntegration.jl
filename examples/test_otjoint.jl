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
using Distances
using JSON

files = readdir(joinpath(@__DIR__), join = true)
json_files = sort(filter(endswith("json"), files))
csv_files = sort(filter(endswith("csv"), files))

# +
function test_ot_joint(csv_file)

    data = DataFrame(CSV.File(csv_file))

    X = Matrix(data[!, ["X1", "X2", "X3"]])
    Y = Vector(data.Y)
    Z = Vector(data.Z)
    database = data.database

    instance = Instance(database, X, Y, Z, Hamming())

    lambda = 0.392
    alpha = 0.714
    percent_closest = 0.2

    sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)
    compute_pred_error!(sol, instance, false)
    return sol.errorpredavg

end
# -

errorpredavg = Float64[]
p = Float64[]
eps = Float64[]
nA = Int[]
nB = Int[]
mB = Vector{Float64}[]

# +
for (csv_file, json_file) in zip(csv_files, json_files)

    params = JSON.parsefile(json_file)

    push!(p, params["p"])
    push!(eps, params["eps"])
    push!(nA, params["nA"])
    push!(nB, params["nB"])
    push!(mB, params["mB"])
    push!(errorpredavg, test_ot_joint(csv_file))
    println("nA = $(nA[end]), nB = $(nB[end])")
    println(
        "p = $(p[end]), eps = $(eps[end]), mB = $(mB[end]), est = $(1 - errorpredavg[end])",
    )

end
# -

df = DataFrame(nA = nA, nB = nB, mB = mB, eps = eps, p = p, errorpred = errorpredavg)

CSV.write(joinpath("results_M5max.csv"), df)

using OTRecod
using DataFrames
using CSV
using Distances
using JSON

files = readdir(joinpath(@__DIR__, "datasets"), join = true) 
json_files = sort(filter( endswith("json"), files))
csv_files = sort(filter( endswith("csv"), files))

function test_ot_joint( csv_file )

    data = DataFrame(CSV.File(csv_file))
    
    X = Matrix(data[!, ["X1", "X2", "X3"]])
    Y = Vector(data.Y)
    Z = Vector(data.Z)
    database = data.database

    dist_choice = Hamming()
    
    instance = Instance( database, X, Y, Z, dist_choice)
    
    lambda_reg = 0.392
    maxrelax = 0.714
    percent_closest = 0.2
    
    sol = ot_joint(instance, maxrelax, lambda_reg, percent_closest)
    OTRecod.compute_pred_error!(sol, instance, false)
    return sol.errorpredavg

end


errorpredavg = Float64[]
p = Float64[]
eps = Float64[]
nA = Int[]
nB = Int[]
mB = Vector{Float64}[]

for (csv_file, json_file) in zip(csv_files, json_files)
    
    params = JSON.parsefile(json_file)
    
    push!(p, params["p"])
    push!(eps, params["eps"])
    push!(nA, params["nA"])
    push!(nB, params["nB"])
    push!(mB, params["mB"])
    push!(errorpredavg, test_ot_joint(csv_file))
    println("nA = $(nA[end]), nB = $(nB[end])")
    println("p = $(p[end]), eps = $(eps[end]), mB = $(mB[end]), est = $(1 - errorpredavg[end])")

end

df = DataFrame( nA = nA, nB = nB, mB = mB, eps = eps, p = p, errorpred = errorpredavg ) 

CSV.write(joinpath("results_M5max.csv"), df)

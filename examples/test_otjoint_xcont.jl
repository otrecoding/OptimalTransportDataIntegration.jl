using OTRecod
using DataFrames
using CSV
using Distances
using JSON
using Statistics

files = readdir(joinpath(@__DIR__, "datasets"), join = true)
json_files = sort(filter( endswith("json"), files))
csv_files = sort(filter( endswith("csv"), files))


# +
function to_categorical(x)
    sort(unique(x)) .== reshape(x, (1, size(x)...))
end

digitize(x, bins) = searchsortedlast.(Ref(bins), x)

function categorize_using_quartile(data)
    
    XA = subset(data, :database => x -> x .== 1.0)
    XB = subset(data, :database => x -> x .== 2.0)

    b1 = quantile(XA.X1, [.25, .5, .75])
    bins11 = vcat(minimum(XA.X1)-100, b1, maximum(XA.X1)+100)

    X11 = digitize(XA.X1, bins11)
    X21 = digitize(XB.X1, bins11)
    
    b1 = quantile(XA.X2, [.25, .5, .75])
    bins12 = vcat(minimum(XA.X2)-100, b1, maximum(XA.X2)+100)
    
    X12 = digitize(XA.X2, bins12)
    X22 = digitize(XB.X2, bins12)
    
    b1 = quantile(XA.X3, [.25, .5, .75])
    bins13 = vcat(minimum(XA.X3)-100, b1, maximum(XA.X3)+100)
    
    X13 = digitize(XA.X3, bins13)
    X23 = digitize(XB.X3, bins13)

    X1 = vcat(X11,X21) .- 1
    X2 = vcat(X12,X22) .- 1
    X3 = vcat(X13,X23) .- 1

    X1c = to_categorical(X1)
    X2c = to_categorical(X2)
    X3c = to_categorical(X3)

    hcat(X1c', X2c', X3c')
        
end


# +
function compute_pred_error_with_otjoint(csv_file)
    
    data = DataFrame(CSV.File(csv_file))

    X = categorize_using_quartile(data)
    Y = Vector(data.Y)
    Z = Vector(data.Z)
    
    database = data.database
    dist_choice = Euclidean()

    instance = Instance( database, X, Y, Z, dist_choice)

    lambda_reg = 0.0
    maxrelax = 0.0
    percent_closest = 0.2

    sol = ot_joint(instance, maxrelax, lambda_reg, percent_closest)
    OTRecod.compute_pred_error!(sol, instance, false)
    return sol.errorpredavg
    
end
# -

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
    push!(errorpredavg, compute_pred_error_with_otjoint(csv_file))
    println("nA = $(nA[end]), nB = $(nB[end])")
    println("p = $(p[end]), eps = $(eps[end]), mB = $(mB[end]), est = $(1 - errorpredavg[end])")

end

df = DataFrame( nA = nA, nB = nB, mB = mB, eps = eps, p = p, errorpred = errorpredavg )

CSV.write(joinpath("results_M5max.csv"), df)

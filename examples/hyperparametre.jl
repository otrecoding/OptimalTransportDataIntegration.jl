using OTRecod
using DataFrames
using CSV
using Distances
using DelimitedFiles
using Statistics

function hyperparam(subdir, csv_files, maxrelax, lambda_reg)

    prederrors = Float64[]

    for csv_file in csv_files
    	@show csv_file
        data = DataFrame(CSV.File(joinpath(subdir, csv_file)))
        
        X = Matrix(data[!, ["X1", "X2", "X3"]])
        Y = Vector(data.Y)
        Z = Vector(data.Z)
        database = data.database
        
        instance = Instance( database, X, Y, Z, Hamming())
        
        percent_closest = 0.2

        for m in maxrelax, l in lambda_reg

            sol = ot_joint(instance, m, l, percent_closest)
            OTRecod.compute_pred_error!(sol, instance, false)
            @show sol.errorpredavg
            push!(prederrors, sol.errorpredavg)

        end

    end

    return prederrors
        
end

maxrelax = collect(0:0.1:2)
lambda_reg = collect(0:0.1:1)

subdir = joinpath(@__DIR__, "datasets")
csv_files = filter( endswith("csv"), readdir(subdir, join = true))

function compute_p_and_m( csv_files)
    pmax = 0
    mmax = 0
    for csv_file in csv_files
        pmax = max(pmax, parse(Int,split(csv_file, '_')[3]) + 1)
        mmax = max(mmax, parse(Int,split(csv_file, '_')[4][1:2]) + 1)
    end
    pmax, mmax
end

@time results = hyperparam(subdir, csv_files, maxrelax, lambda_reg)

open("results.txt", "w") do io
    writedlm(io, results)
end

@show pmax, mmax = compute_p_and_m(csv_files)
k = 0
errors = zeros(length(maxrelax), length(lambda_reg), pmax)
for csv_file in csv_files
    p = parse(Int,split(csv_file, '_')[3]) + 1
    for i in eachindex(maxrelax), j in eachindex(lambda_reg)
        global k
        k += 1
        errors[i,j,p] += results[k] / mmax
    end
end

@show pmax
for p in 1:pmax
    x = errors[:,:,p]
    i, j = Tuple(findmin(x)[2])
    println(" maxrelax = $(maxrelax[i]) lambda_reg = $(lambda_reg[j])")
end

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Julia 1.11.2
#     language: julia
#     name: julia-1.11
# ---

using OptimalTransportDataIntegration
import CSV
using DataFrames
using JuMP, Clp
import .Iterators: product
import Distances: pairwise, Hamming, Cityblock

onecold(X) = map(argmax, eachrow(X))

function loss_crossentropy(Y, F)
    ϵ = 1e-12
    res = zeros(size(Y, 1), size(F, 1))
    logF = log.(F .+ ϵ)
    for i in axes(Y, 2)
        res .+= -Y[:, i] .* logF[:, i]'
    end
    return res
end

"""
    optimal_modality(values, loss, weight)

- values: vector of possible values
- weight: vector of weights 
- loss: matrix of size len(Weight) * len(Values)

- Returns an argmin over value in values of the scalar product <loss[value,],weight> 
"""
function optimal_modality(values, loss, weight)

    cost_for_each_modality = Float64[]
    for j in eachindex(values)
        s = 0
        for i in axes(loss, 1)
            s += loss[i, j] * weight[i]
        end
        push!(cost_for_each_modality, s)
    end

    return values[argmin(cost_for_each_modality)]

end


# -

function unbalanced_solver(data; lambda_reg = 0.392, maxrelax = 0.714)

    T = Int
    Ylevels = 1:4
    Zlevels = 1:3
    
    X = Matrix{Int}(one_hot_encoder(data[!, [:X1, :X2, :X3]]))
    Xlevels = sort(unique(eachrow(X)))
    Y = Vector{T}(data.Y)
    Z = Vector{T}(data.Z)
    
    
    nx = size(X, 2)
    XYA = Vector{Int}[]
    XZB = Vector{Int}[]
    for (y, x) in product(Ylevels, Xlevels)
        push!(XYA, [x...; y])
    end
    for (z, x) in product(Zlevels, Xlevels)
        push!(XZB, [x...; z])
    end
    
    XA_hot = stack([v[1:nx] for v in XYA], dims = 1)
    XB_hot = stack([v[1:nx] for v in XZB], dims = 1)
    
    C0 = pairwise(Hamming(), XA_hot, XB_hot; dims = 1)
    
    c = Dict()
    
    for (i, (y,x1)) in enumerate(product(Ylevels, eachindex(Xlevels)))
        for (j, (z,x2)) in enumerate(product(Zlevels, eachindex(Xlevels)))
            c[(x1, y, x2, z)] = C0[i,j]
        end
    end
    
    c0 = zeros(Int32, length(product(Ylevels, Xlevels)), length(product(Zlevels, Xlevels)))
    for p1 in axes(c0,1)
        for p2 in axes(c0,2)
            x1 = findfirst( ==(XYA[p1][1:nx]), Xlevels)
            y = last(XYA[p1])
            x2 = findfirst( ==(XZB[p2][1:nx]), Xlevels)
            z = last(XZB[p2])
            c0[p1,p2] = c[(x1, y, x2, z)]
            end
        end
        
    @assert c0 ≈ C0
    
    
    
    Xobserv = Matrix(data[!, [:X1, :X2, :X3]])
    Xvalues = unique(eachrow(Xobserv))
    dist_X = pairwise(Cityblock(), Xvalues, Xvalues)
    voisins_X = dist_X .<= 1
    
    
    base = data.database
    
    indA = findall(base .== 1)
    indB = findall(base .== 2)
    
    
    YA = view(Y, indA)
    YB = view(Y, indB)
    ZA = view(Z, indA)
    ZB = view(Z, indB)
    
    XA = view(X, indA, :)
    XB = view(X, indB, :)
    
    YA_hot = one_hot_encoder(YA, Ylevels)
    ZA_hot = one_hot_encoder(ZA, Zlevels)
    YB_hot = one_hot_encoder(YB, Ylevels)
    ZB_hot = one_hot_encoder(ZB, Zlevels)
    
    XYA = hcat(XA, YA)
    XZB = hcat(XB, ZB)
    
    distance = Hamming()
    
    # Compute data for aggregation of the individuals
    
    nA = length(indA)
    nB = length(indB)
    
    # Compute the indexes of individuals with same covariates
    indXA = Dict{T,Array{T}}()
    indXB = Dict{T,Array{T}}()
    Xlevels = sort(unique(eachrow(X)))
    # -
    
    for (i, x) in enumerate(Xlevels)
        distA = vec(pairwise(distance, x[:, :], XA', dims = 2))
        distB = vec(pairwise(distance, x[:, :], XB', dims = 2))
        indXA[i] = findall(distA .< 0.1)
        indXB[i] = findall(distB .< 0.1)
    end
    
    Ylevels_hot = one_hot_encoder(Ylevels)
    Zlevels_hot = one_hot_encoder(Zlevels)
    
    # +
    Yloss = loss_crossentropy(YA_hot, Ylevels_hot)
    Zloss = loss_crossentropy(ZB_hot, Zlevels_hot)
    @show alpha1 = 1 / maximum(loss_crossentropy(Ylevels_hot, Ylevels_hot))
    @show alpha2 = 1 / maximum(loss_crossentropy(Zlevels_hot, Zlevels_hot))
    
    
    ## Optimal Transport
    
    C0 = pairwise(Hamming(), XA_hot, XB_hot; dims = 1) 
    C = C0 ./ maximum(C0)
    
    
    est_opt = 0.0
    
    YBpred = zeros(T, nB)
    ZApred = zeros(T, nA)
    
    # Compute the estimators that appear in the model
    
    estim_XA = Dict([(x, length(indXA[x]) / nA) for x = eachindex(indXA)])
    estim_XB = Dict([(x, length(indXB[x]) / nB) for x = eachindex(indXA)])
    estim_XA_YA = Dict([
        ((x, y), length(indXA[x][findall(YA[indXA[x]] .== y)]) / nA) 
             for x = eachindex(indXA), y in Ylevels
    ])
    estim_XB_ZB = Dict([
        ((x, z), length(indXB[x][findall(ZB[indXB[x]] .== z)]) / nB) 
             for x = eachindex(indXB), z in Zlevels
    ])
    
    @show nbX = length(indXA)
    @show length(Xlevels)
    @show indXA
    
    model = Model(Clp.Optimizer)
    set_optimizer_attribute(model, "LogLevel", 0)
    
    @variable(model,
            gamma[x1 in 1:nbX, 
                  y in Ylevels, 
                  x2 in 1:nbX,
                  z in Zlevels] >= 0,
            base_name = "gamma"
        )
    
          
    @variable(model, error_XY[x1 in 1:nbX, y in Ylevels], base_name = "error_XY")
    
    @variable(model,
            abserror_XY[x1 in 1:nbX, y in Ylevels] >= 0,
            base_name = "abserror_XY"
    )
    @variable(model, error_XZ[x2 in 1:nbX, z in Zlevels], base_name = "error_XZ")
    
    @variable(model,
            abserror_XZ[x2 in 1:nbX, z in Zlevels] >= 0,
            base_name = "abserror_XZ"
    )
    
    # - assign sufficient probability to each class of covariates with the same outcome
    @constraint(model,
        ctYandXinA[x1 in 1:nbX, y in Ylevels],
        sum(gamma[x1, y, x2, z] for x2 in 1:nbX, z in Zlevels) == estim_XA_YA[x1, y] + error_XY[x1, y]
    )
    
    
    # - we impose that the probability of Y conditional to X is the same in the two databases
    # - the consequence is that the probability of Y and Z conditional to Y is also the same in the two bases
    @constraint(model,
        ctZandXinA[x2 in 1:nbX, z in Zlevels],
        sum(gamma[x1, y, x2, z] for x1 in 1:nbX, y in Ylevels) ==
        estim_XB_ZB[x2, z] + error_XZ[x2, z]
    )
    
    
    # - recover the norm 1 of the error
    @constraint(model, [x1 in 1:nbX, y in Ylevels], error_XY[x1, y] <= abserror_XY[x1, y])
    @constraint(model, [x1 in 1:nbX, y in Ylevels], -error_XY[x1, y] <= abserror_XY[x1, y])
    @constraint(model, sum(abserror_XY[x1, y] for x1 = 1:nbX, y in Ylevels) <= maxrelax / 2.0)
    @constraint(model, sum(error_XY[x1, y] for x1 = 1:nbX, y in Ylevels) == 0.0)
    @constraint(model, [x2 in 1:nbX, z in Zlevels], error_XZ[x2, z] <= abserror_XZ[x2, z])
    @constraint(model, [x2 in 1:nbX, z in Zlevels], -error_XZ[x2, z] <= abserror_XZ[x2, z])
    @constraint(model, sum(abserror_XZ[x2, z] for x2 = 1:nbX, z in Zlevels) <= maxrelax / 2.0)
    @constraint(model, sum(error_XZ[x2, z] for x2 = 1:nbX, z in Zlevels) == 0.0)

    # - regularization
    @variable(
        model,
        reg_absA[
            x1 in 1:nbX,
            x2 in findall(voisins_X[x1, :]),
            y in Ylevels,
            z in Zlevels,
        ] >= 0
    )
    @constraint(
        model,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), y in Ylevels, x in 1:nbX, z in Zlevels],
        reg_absA[x1, x2, y, z] >=
        gamma[x1, y, x,z] / (max(1, length(indXA[x1])) / nA) -
        gamma[x2, y, x,z] / (max(1, length(indXA[x2])) / nA)
    )
    @constraint(
        model,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), y in Ylevels, x in 1:nbX, z in Zlevels],
        reg_absA[x1, x2, y, z] >=
        gamma[x2, y,x, z] / (max(1, length(indXA[x2])) / nA) -
        gamma[x1, y, x,z] / (max(1, length(indXA[x1])) / nA)
    )
    @expression(
        model,
        regtermA,
        sum(
            1 / length(voisins_X[x1, :]) * reg_absA[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        )
    )

 # - regularization
    @variable(
        model,
        reg_absB[
            x1 in 1:nbX,
            x2 in findall(voisins_X[x1, :]),
            y in Ylevels,
            z in Zlevels,
        ] >= 0
    )
    @constraint(
        model,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), x in 1:nbX, y in Ylevels, z in Zlevels],
        reg_absB[x1, x2, y, z] >=
        gamma[x, y, x1,z] / (max(1, length(indXB[x1])) / nB) -
        gamma[x, y, x2,z] / (max(1, length(indXB[x2])) / nB)
    )
    @constraint(
        model,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), x in 1:nbX, y in Ylevels, z in Zlevels],
        reg_absB[x1, x2, y, z] >=
        gamma[x, y,x2, z] / (max(1, length(indXB[x2])) / nB) -
        gamma[x, y, x1,z] / (max(1, length(indXB[x1])) / nB)
    )
    @expression(
        model,
        regtermB,
        sum(
            1 / length(voisins_X[x1, :]) * reg_absB[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        )
    )

    
    # by default, the OT cost and regularization term are weighted to lie in the same interval
    @objective(
        model,
        Min,
        sum(c[x1,y, x2,z] * gamma[x1, y,x2, z] for y in Ylevels, z in Zlevels, 
            x1 = 1:nbX, x2 = 1:nbX) +
        lambda_reg * sum(
            1 / length(voisins_X[x1, :]) * reg_absA[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        ) +
        lambda_reg * sum(
            1 / length(voisins_X[x1, :]) * reg_absB[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        )
    )

    # Solve the problem
    optimize!(model)

    # Extract the values of the solution
    gamma_val = [value(gamma[x1, y,x2, z]) for x1 = 1:nbX, y in Ylevels,x2 = 1:nbX, z in Zlevels]

#=

    for j in eachindex(yB_pred)
        yB_pred[j] = optimal_modality(Ylevels, Yloss, view(G, :, j))
    end

    for i in eachindex(zA_pred)
        zA_pred[i] = optimal_modality(Zlevels, Zloss, view(G, i, :))
    end

    YBpred_hot = one_hot_encoder(yB_pred, Ylevels)
    ZApred_hot = one_hot_encoder(zA_pred, Zlevels)

    ### Update Cost matrix

    cost1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
    cost2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
    fcost = cost1 .+ cost2'

    C .= C0 ./ maximum(C0) .+ fcost

    zA_pred_hot_i = zeros(T, (nA, length(Zlevels)))
    yB_pred_hot_i = zeros(T, (nB, length(Ylevels)))

    for i in axes(XYA, 1)
        ind = findfirst(XYA[i, :] == v for v in XYA2)
        zA_pred_hot_i[i, :] .= zA_pred_hot[ind, :]
    end

    for i in axes(XZB, 1)
        ind = findfirst(XZB[i, :] == v for v in XZB2)
        yB_pred_hot_i[i, :] .= yB_pred_hot[ind, :]
    end

    YBpred .= onecold(yB_pred_hot_i)
    ZApred .= onecold(zA_pred_hot_i)

    est = (sum(YB .== YBpred) .+ sum(ZA .== ZApred)) ./ (nA + nB)

    est_opt = max(est_opt, est)

=#
end

csv_file = joinpath("dataset.csv")

data = CSV.read(csv_file, DataFrame)

@time unbalanced_solver(data, maxrelax = 0.01)

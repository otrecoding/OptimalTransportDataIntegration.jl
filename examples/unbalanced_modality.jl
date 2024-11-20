# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:light,ipynb
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

# +
using OptimalTransportDataIntegration
import OptimalTransportDataIntegration: Instance
using CSV
using DataFrames
import PythonOT
import .Iterators: product
import Distances: pairwise, Hamming


function main(params, data)

    onecold(X) = map(argmax, eachrow(X))


    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    X = Matrix{Int}(one_hot_encoder(data[!, [:X1, :X2, :X3]]))
    Y = Vector{Int}(data.Y)
    Z = Vector{Int}(data.Z)

    YA = Y[indA]
    YB = Y[indB]
    ZA = Z[indA]
    ZB = Z[indB]

    XA = X[indA, :]
    XB = X[indB, :]

    XA_hot = X[indA, :]
    XB_hot = X[indB, :]

    Ylevels = collect(1:4)
    Zlevels = collect(1:3)

    YA_hot = one_hot_encoder(YA, Ylevels)
    ZA_hot = one_hot_encoder(ZA, Zlevels)
    YB_hot = one_hot_encoder(YB, Ylevels)
    ZB_hot = one_hot_encoder(ZB, Zlevels)

    XA_hot_i = copy(XA_hot)
    XB_hot_i = copy(XB_hot)
    yA_i = YA
    zB_i = ZB

    nA_i, nB_i = size(XA_hot_i, 1), size(XB_hot_i, 1)

    XYA_i = hcat(XA_hot_i, yA_i)
    XZB_i = hcat(XB_hot_i, zB_i)

    Xhot = one_hot_encoder(Matrix(data[!, ["X1", "X2", "X3"]]))

    Y = Vector(data.Y)
    Z = Vector(data.Z)
    base = data.database

    distance = Hamming()


    # Compute data for aggregation of the individuals


    Xobserv = vcat(Xhot[indA, :], Xhot[indB, :])
    Yobserv = vcat(Y[indA], Y[indB])
    Zobserv = vcat(Z[indA], Z[indB])

    nA = length(indA)
    nB = length(indB)

    # list the distinct modalities in A and B
    indY = Dict((m, findall(Y[indA] .== m)) for m in Ylevels)
    indZ = Dict((m, findall(Z[indB] .== m)) for m in Zlevels)

    # compute the distance between pairs of individuals in different bases
    # devectorize all the computations to go about twice faster only compute norm 1 here
    a = Xhot[indA, :]'
    b = Xhot[indB, :]'

    # Compute the indexes of individuals with same covariates
    indXA = Dict{Int64,Array{Int64}}()
    indXB = Dict{Int64,Array{Int64}}()
    Xlevels = sort(unique(eachrow(Xhot)))

    nbX = 0
    # aggregate both bases
    for x in Xlevels
        nbX = nbX + 1
        distA = vec(pairwise(distance, x[:, :], a, dims = 2))
        distB = vec(pairwise(distance, x[:, :], b, dims = 2))
        indXA[nbX] = findall(distA .< 0.1)
        indXB[nbX] = findall(distB .< 0.1)
    end

    @show sort(unique(Y)), sort(unique(Z))

    nbX = length(indXA)

    wa = vec([
        length(indXA[x][findall(Y[indXA[x]] .== y)]) / params.nA for y in Ylevels, x = 1:nbX
    ])
    wb = vec([
        length(indXB[x][findall(Z[indXB[x].+params.nA] .== z)]) / params.nB for
        z in Zlevels, x = 1:nbX
    ])

    @show wa
    @show wb

    wa2 = filter(>(0), wa)
    wb2 = filter(>(0), wb)

    # -


    @show length(sort(Xlevels))

    XYA = Vector{Int}[]
    XZB = Vector{Int}[]
    for (y, x) in product(Ylevels, Xlevels)
        push!(XYA, [x...; y])
    end
    for (z, x) in product(Zlevels, Xlevels)
        push!(XZB, [x...; z])
    end

    XYA2 = XYA[wa.>0] ### XYA observés
    XZB2 = XZB[wb.>0] ### XZB observés

    # +
    Y_hot = one_hot_encoder(Ylevels)
    Z_hot = one_hot_encoder(Zlevels)

    nx = size(Xobserv, 2) ## Nb modalités x 

    # les x parmi les XYA observés, potentiellement des valeurs repetées 
    XA_hot = stack([v[1:nx] for v in XYA2], dims = 1)
    # les x dans XZB observés, potentiellement des valeurs repetées 
    XB_hot = stack([v[1:nx] for v in XZB2], dims = 1)

    yA = getindex.(XYA2, nx + 1)  ## les y  parmi les XYA observés, des valeurs repetées 
    @show yA
    yA_hot = one_hot_encoder(yA, Ylevels)
    zB = getindex.(XZB2, nx + 1) # les z dans XZB observés, potentiellement des valeurs repetées 
    @show zB
    zB_hot = one_hot_encoder(zB, Zlevels)

    # +
    # ## Algorithm
    #
    # ### Initialisation 
    # +
    nA = size(XYA2, 1) # number of observed different values in A
    nB = size(XZB2, 1) # number of observed different values in B
    nbrvarX = 3

    dimXZB = length(XZB2[1])
    dimXYA = length(XYA2[1])

    NumberOfIterations = 3

    yB_pred = zeros(nB)
    zA_pred = zeros(nA)
    function loss_crossentropy(Y, F)
        ϵ = 1e-12
        res = zeros(size(Y, 1), size(F, 1))
        logF = log.(F .+ ϵ)
        for i in axes(Y, 2)
            res .+= -Y[:, i] .* logF[:, i]'
        end
        return res
    end
    Y_loss = loss_crossentropy(yA_hot, Y_hot)
    Z_loss = loss_crossentropy(zB_hot, Z_hot)

    # +
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

    # Zc = onehot(possible values of Z)
    # zB2 = onehot(zB), où zB est les z dans XZB observés, 
    # potentiellement des valeurs repetées 

    # ### Optimal Transport

    # +
    C0 = pairwise(Hamming(), XA_hot, XB_hot; dims = 1) .* nx ./ nbrvarX
    C = C0 ./ maximum(C0)

    zA_pred_hot_i = zeros(Int, (nA_i, length(Zlevels)))
    yB_pred_hot_i = zeros(Int, (nB_i, length(Ylevels)))

    # +
    NumberOfIterations = 10

    for iter = 1:NumberOfIterations

        G = PythonOT.mm_unbalanced(wa2, wb2, C, 0.1; div = "kl") #unbalanced

        for j in eachindex(yB_pred)
            yB_pred[j] = optimal_modality(Ylevels, Y_loss, G[:, j])
        end


        yB_pred_hot = one_hot_encoder(yB_pred, Ylevels)

        ### Compute best g: XxY-->Z

        for i in eachindex(zA_pred)
            zA_pred[i] = optimal_modality(Zlevels, Z_loss, G[i, :])
        end


        zA_pred_hot = one_hot_encoder(zA_pred, Zlevels)

        ### Update Cost matrix
        alpha1 = 1 / maximum(loss_crossentropy(yA_hot, yB_pred_hot))
        alpha2 = 1 / maximum(loss_crossentropy(zB_hot, zA_pred_hot))

        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
        fcost = chinge1 .+ chinge2'


        C .= C0 ./ maximum(C0) .+ fcost


        ### Predict

        for i in axes(XYA_i, 1)
            ind = findfirst(XYA_i[i, :] == v for v in XYA2)
            if isnothing(ind)
                zA_pred_hot_i[i, :] .= 0
            else
                zA_pred_hot_i[i, :] .= zA_pred_hot[ind, :]
            end
        end

        for i in axes(XZB_i, 1)
            ind = findfirst(XZB_i[i, :] == v for v in XZB2)
            if isnothing(ind)
                yB_pred_hot_i[i, :] .= 0
            else
                yB_pred_hot_i[i, :] .= yB_pred_hot[ind, :]
            end
        end


        YBpred = onecold(yB_pred_hot_i)
        ZApred = onecold(zA_pred_hot_i)

        ### Evaluate 

        est = (sum(YB .== YBpred) .+ sum(ZA .== ZApred)) ./ (nA_i + nB_i)
        println("est $(sum(YB .== YBpred)/nB_i) $(sum(ZA .== ZApred)/nA_i)")

    end
    # -


end

params = DataParameters(nA = 1000, nB = 1000, mB = [2, 0, 0], eps = 0.0, p = 0.2)
# data = generate_xcat_ycat(params)
data = CSV.read(joinpath(@__DIR__, "data_good.csv"), DataFrame)
@show sort(unique(data.Y)), sort(unique(data.Z))
@time main(params, data)

params = DataParameters(nA = 1000, nB = 1000, mB = [0, 0, 0], eps = 0.0, p = 0.2)
data = CSV.read(joinpath(@__DIR__, "data_bad.csv"), DataFrame)
@show sort(unique(data.Y)), sort(unique(data.Z))

@time main(params, data)

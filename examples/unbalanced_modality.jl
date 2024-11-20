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


function main()

    # params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0.0, p=0.2)
    params = DataParameters(nA = 1000, nB = 1000, mB = [0, 0, 0], eps = 0.0, p = 0.2)
    # data = generate_xcat_ycat(params)
    data = CSV.read(joinpath(@__DIR__, "data_bad.csv"), DataFrame)
    @show sort(unique(data.Y)), sort(unique(data.Z))

    onecold(X) = map(argmax, eachrow(X))

    Xnames_hot, X_hot, Y_hot, Z_hot, XA_hot, YA_hot, XB_hot, ZB_hot, YB_true, ZA_true =
        prep_data(data)

    # données individuelles annexées par i

    XA_hot_i = copy(XA_hot)
    XB_hot_i = copy(XB_hot)
    yA_i = onecold(YA_hot)
    zB_i = onecold(ZB_hot)

    nA_i, nB_i = size(XA_hot_i, 1), size(XB_hot_i, 1)

    XYA_i = hcat(XA_hot_i, yA_i)
    XZB_i = hcat(XB_hot_i, zB_i)

    Xhot = one_hot_encoder(Matrix(data[!, ["X1", "X2", "X3"]]))

    Y = Vector(data.Y)
    Z = Vector(data.Z)
    database = data.database

    dist_choice = Hamming()

    Xlevels = Vector{Int}[]
    for i in (0, 1), j in (0, 1, 2), k in (0, 1, 2, 3)
        push!(Xlevels, [i; j == 1; j == 2; k == 1; k == 2; k == 3])
    end
    Ylevels = collect(1:4)
    Zlevels = collect(1:3)

    # # Compute data for aggregation of the individuals

    instance = Instance(database, Xhot, Y, Ylevels, Z, Zlevels, dist_choice)

    indXA = instance.indXA
    indXB = instance.indXB

    @show sort(unique(Y)), sort(unique(Z))

    # -


    # +
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

    nx = size(instance.Xobserv, 2) ## Nb modalités x 

    XA_hot = stack([v[1:nx] for v in XYA2], dims = 1) # les x parmi les XYA observés, potentiellement des valeurs repetées 
    XB_hot = stack([v[1:nx] for v in XZB2], dims = 1) # les x dans XZB observés, potentiellement des valeurs repetées 

    yA = getindex.(XYA2, nx + 1)  ## les y  parmi les XYA observés, des valeurs repetées 
    yA_hot = one_hot_encoder(yA, Ylevels)
    zB = getindex.(XZB2, nx + 1) # les z dans XZB observés, potentiellement des valeurs repetées 
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


        YB_pred = onecold(yB_pred_hot_i)
        ZA_pred = onecold(zA_pred_hot_i)

        ### Evaluate 

        est = (sum(YB_true .== YB_pred) .+ sum(ZA_true .== ZA_pred)) ./ (nA_i + nB_i)
        println("est $(sum(YB_true .== YB_pred)/nB_i) $(sum(ZA_true .== ZA_pred)/nA_i)")

    end
    # -


end

@time main()

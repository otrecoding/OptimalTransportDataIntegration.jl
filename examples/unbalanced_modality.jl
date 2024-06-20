# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Julia 1.10.4
#     language: julia
#     name: julia-1.10
# ---

# +
using OptimalTransportDataIntegration
using OTRecod
using CSV
using DataFrames
import PythonOT
import .Iterators: product
import Distances: pairwise
import Distances: Hamming
import Flux
using ProgressMeter



onecold(X) = map(argmax, eachrow(X))
    
function loss_crossentropy(Y, F)
    ϵ = 1e-12
    res = zeros(size(Y,1), size(F,1))
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
    
    cost_for_each_modality=Float64[]
    for j in eachindex(values)
        s=0
        for i in axes(loss, 1)
            s += loss[i,j] * weight[i]
        end
        push!(cost_for_each_modality, s)
    end
        
    return values[argmin(cost_for_each_modality)]

end

# +
function unbalanced_modality( data; iterations = 1)
    
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    nA = size(dba, 1)
    nB = size(dbb, 1)

    Xnames_hot, X_hot, Y, Z, XA_hot, YA, XB_hot, ZB, YB_true, ZA_true = prep_data(data)


    XA_hot_i = copy(XA_hot)
    XB_hot_i = copy(XB_hot)
    yA_i  = onecold(YA)
    zB_i  = onecold(ZB)

    nA_i, nB_i  = size(XA_hot_i, 1), size(XB_hot_i, 1)

    XYA_i = hcat(XA_hot_i, yA_i)
    XZB_i = hcat(XB_hot_i, zB_i)

    Xhot = one_hot_encoder(Matrix(data[!, ["X1", "X2", "X3"]]))

    Y = Vector(data.Y)
    Z = Vector(data.Z)
    database = data.database

    dist_choice = Hamming()
    
    instance = OTRecod.Instance( database, Xhot, Y, Z, dist_choice)
    

    # Compute data for aggregation of the individuals

    indXA = instance.indXA
    indXB = instance.indXB
    nbX = length(indXA)

    wa = vec([length(indXA[x][findall(instance.Yobserv[indXA[x]] .== y)]) / nA for y in instance.Y, x = 1:nbX])
    wb = vec([length(indXB[x][findall(instance.Zobserv[indXB[x] .+ nA] .== z)]) / nB for z in instance.Z, x = 1:nbX ])  
    wa2 = wa[wa .> 0.0]
    wb2 = wb[wb .> 0.0]

    Xvalues = stack(sort(unique(eachrow(one_hot_encoder(instance.Xval)))), dims=1)

    Xobserv = sort(unique(eachrow(instance.Xobserv)))
    Yobserv = sort(unique(instance.Yobserv))
    Zobserv = sort(unique(instance.Zobserv))

    XYA = Vector{Int}[]
    XZB = Vector{Int}[]
    for (y,x) in product(Yobserv,Xobserv)
        push!(XYA, [x...; y])
    end
    for (z,x) in product(Zobserv,Xobserv)
        push!(XZB, [x...; z])
    end

    XYA2 = XYA[wa .> 0] ### XYA observés
    XZB2 = XZB[wb .> 0] ### XZB observés

    Y_hot = one_hot_encoder(instance.Y)
    Z_hot = one_hot_encoder(instance.Z)

    nx = size(Xvalues, 2) ## Nb modalités x 

    XA_hot = stack([v[1:nx] for v in XYA2], dims=1) # les x parmi les XYA observés, potentiellement des valeurs repetées 
    XB_hot = stack([v[1:nx] for v in XZB2], dims=1) # les x dans XZB observés, potentiellement des valeurs repetées 

    yA = getindex.(XYA2, nx+1)  ## les y  parmi les XYA observés, des valeurs repetées 
    yA_hot = one_hot_encoder(yA)
    zB = getindex.(XZB2, nx+1) # les z dans XZB observés, potentiellement des valeurs repetées 
    zB_hot = one_hot_encoder(zB)

    # ## Algorithm
    #
    # ### Initialisation 

    nA = size(XYA2, 1) # number of observed different values in A
    nB = size(XZB2, 1) # number of observed different values in B
    nbrvarX = 3

    dimXZB = length(XZB2[1])
    dimXYA = length(XYA2[1])

    yB_pred = zeros(nB)
    zA_pred = zeros(nA)
    
    Y_loss = loss_crossentropy(yA_hot, Y_hot)
    Z_loss = loss_crossentropy(zB_hot, Z_hot) 

    ### Optimal Transport

    C0 = pairwise(Hamming(), XA_hot, XB_hot; dims=1) .* nx ./ nbrvarX
    C = C0 ./ maximum(C0)

    zA_pred_hot_i = zeros(Int, (nA_i,length(instance.Z)))
    yB_pred_hot_i = zeros(Int, (nB_i,length(instance.Y)))

    est = 0.0

    for iter in 1:iterations
    
        G = PythonOT.mm_unbalanced(wa2, wb2, C, 0.1; div="kl") #unbalanced
    
        for j in eachindex(yB_pred)
             yB_pred[j] = optimal_modality(instance.Y, Y_loss, G[:,j])
        end
        for i in eachindex(zA_pred)
            zA_pred[i] = optimal_modality(instance.Z, Z_loss, G[i,:])
        end
    
        yB_pred_hot = one_hot_encoder(yB_pred)
        zA_pred_hot = one_hot_encoder(zA_pred)
 
        ### Update Cost matrix
        alpha1 = 1 / maximum(loss_crossentropy(yA_hot, yB_pred_hot))
        alpha2 = 1 / maximum(loss_crossentropy(zB_hot, zA_pred_hot))
 
        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
        fcost = chinge1 .+ chinge2'
 
        C .= C0 ./ maximum(C0) .+ fcost
 
        ### Predict
    
        for i in axes(XYA_i, 1)
            ind = findfirst(XYA_i[i,:] == v for v in XYA2)
            zA_pred_hot_i[i,:] .= zA_pred_hot[ind,:]
        end

        for i in axes(XZB_i, 1)
            ind = findfirst(XZB_i[i,:] == v for v in XZB2)
            yB_pred_hot_i[i,:] .= yB_pred_hot[ind,:]
        end
 
        YB_pred = onecold(yB_pred_hot_i) 
        ZA_pred = onecold(zA_pred_hot_i)
 
        ### Evaluate 
 
        est = (sum(YB_true .== YB_pred) .+ sum(ZA_true .== ZA_pred)) ./ (nA_i + nB_i)

    end

    return est

end
# -

function run_simulations( simulations )

    params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)
    prediction_quality = Float64[]
    simulations = 100
    for i in 1:simulations
       data = generate_xcat_ycat(params)
       push!(prediction_quality, unbalanced_modality(data))
    end

    prediction_quality

end

run_simulations( 100 )

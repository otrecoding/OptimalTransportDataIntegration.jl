# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl,ipynb
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
import Distances: pairwise

params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)
params = DataParameters(nA=1000, nB=1000, mB=[0, 0, 0], eps=0.00, p=0.2)
data = generate_xcat_ycat(params)
# data = CSV.read(joinpath(@__DIR__, "data.csv"), DataFrame)
@show sort(unique(data.Y)), sort(unique(data.Z))

dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

onecold(X) = map(argmax, eachrow(X))

Xnames_hot, X_hot, Y, Z, XA_hot, YA, XB_hot, ZB, YB_true, ZA_true = prep_data(data)
Xnames_hot
# -

# jdonnées individuelles annexées par i

# +
XA_hot_i = copy(XA_hot)
XB_hot_i = copy(XB_hot)
yA_i  = onecold(YA)
zB_i  = onecold(ZB)

nA_i, nB_i  = size(XA_hot_i, 1), size(XB_hot_i, 1)
# -

XYA_i = hcat(XA_hot_i, yA_i)
XZB_i = hcat(XB_hot_i, zB_i)

# +
import Distances: Hamming

Xhot = one_hot_encoder(Matrix(data[!, ["X1", "X2", "X3"]]))


# +
Y = Vector(data.Y)
Z = Vector(data.Z)
database = data.database

dist_choice = Hamming()

Ylevels = collect(1:4)
Zlevels = collect(1:3)
    
instance = Instance( database, Xhot, Y, Ylevels, Z, Zlevels, dist_choice)
    
instance.Ylevels, instance.Zlevels
# -

sort(unique(instance.Yobserv)), sort(unique(instance.Zobserv))

# # Compute data for aggregation of the individuals

# +
indXA = instance.indXA
indXB = instance.indXB
Xobserv = instance.Xobserv
Yobserv = instance.Yobserv
Zobserv = instance.Zobserv
nbX = length(indXA)

wa = vec([length(indXA[x][findall(Yobserv[indXA[x]] .== y)]) / params.nA for y in instance.Ylevels, x = 1:nbX])
wb = vec([length(indXB[x][findall(Zobserv[indXB[x] .+ params.nA] .== z)]) / params.nB for z in instance.Zlevels, x = 1:nbX ])  
wa2 = wa[wa .> 0.0]
wb2 = wb[wb .> 0.0]
# -

Xvalues = stack(sort(unique(eachrow(one_hot_encoder(instance.Xval)))), dims=1)


instance.Yobserv

Yobserv = collect(1:4) # sort(unique(instance.Yobserv))
Zobserv = collect(1:3) # sort(unique(instance.Zobserv))

Xobserv = sort(unique(eachrow(instance.Xobserv)))

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

# +
Y_hot = one_hot_encoder(instance.Ylevels)
Z_hot = one_hot_encoder(instance.Zlevels)

nx = size(Xvalues, 2) ## Nb modalités x 

XA_hot = stack([v[1:nx] for v in XYA2], dims=1) # les x parmi les XYA observés, potentiellement des valeurs repetées 
XB_hot = stack([v[1:nx] for v in XZB2], dims=1) # les x dans XZB observés, potentiellement des valeurs repetées 

yA = getindex.(XYA2, nx+1)  ## les y  parmi les XYA observés, des valeurs repetées 
yA_hot = one_hot_encoder(yA, instance.Ylevels)
zB = getindex.(XZB2, nx+1) # les z dans XZB observés, potentiellement des valeurs repetées 
zB_hot = one_hot_encoder(zB, instance.Zlevels)

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
    res = zeros(size(Y,1), size(F,1))
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
# -

# Zc = onehot(possible values of Z)
# zB2 = onehot(zB), où zB est les z dans XZB observés, 
# potentiellement des valeurs repetées 

# ### Optimal Transport

# +
C0 = pairwise(Hamming(), XA_hot, XB_hot; dims=1) .* nx ./ nbrvarX
C = C0 ./ maximum(C0)

zA_pred_hot_i = zeros(Int, (nA_i,length(instance.Zlevels)))
yB_pred_hot_i = zeros(Int, (nB_i,length(instance.Ylevels)))

# +
NumberOfIterations = 10

for iter in 1:NumberOfIterations
    
    G = PythonOT.mm_unbalanced(wa2, wb2, C, 0.1; div="kl") #unbalanced
    

    for j in eachindex(yB_pred)
         yB_pred[j] = optimal_modality(instance.Ylevels, Y_loss, G[:,j])
    end

    
    yB_pred_hot = one_hot_encoder(yB_pred, instance.Ylevels)
     
    ### Compute best g: XxY-->Z
 
    for i in eachindex(zA_pred)
        zA_pred[i] = optimal_modality(instance.Zlevels, Z_loss, G[i,:])
    end
 

    zA_pred_hot = one_hot_encoder(zA_pred, instance.Zlevels)
 
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
        if isnothing(ind)
            zA_pred_hot_i[i,:] .= 0
        else
            zA_pred_hot_i[i,:] .= zA_pred_hot[ind,:]
        end
    end

    for i in axes(XZB_i, 1)
        ind = findfirst(XZB_i[i,:] == v for v in XZB2)
        if isnothing(ind) 
            yB_pred_hot_i[i,:] .= 0
        else
            yB_pred_hot_i[i,:] .= yB_pred_hot[ind,:]
        end
    end
 

    YB_pred = onecold(yB_pred_hot_i) 
    ZA_pred = onecold(zA_pred_hot_i)
 
    ### Evaluate 
 
    est = (sum(YB_true .== YB_pred) .+ sum(ZA_true .== ZA_pred)) ./ (nA_i + nB_i)
    println("est $(sum(YB_true .== YB_pred)/nB_i) $(sum(ZA_true .== ZA_pred)/nA_i)")

end
# -


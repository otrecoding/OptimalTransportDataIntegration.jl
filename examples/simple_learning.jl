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
#     display_name: Julia 1.10.3
#     language: julia
#     name: julia-1.10
# ---

# +
# # Simple learning
# +
using OptimalTransportDataIntegration
using CSV
using DataFrames
import Flux
import Flux: Chain, Dense, softmax, relu, logitcrossentropy

function onehot(x :: AbstractMatrix)
    res = Vector{Float32}[] 
    for col in eachcol(x)
        levels = filter( x -> x != 0, sort(unique(col)))
        for level in levels
            push!(res, col .== level)
        end
    end
    return stack(res, dims=1) 
end

function onehot(x :: AbstractVector)
    levels = filter( x -> x != 0, sort(unique(x)))
    Float32.(permutedims(x) .== levels)
end

function onecold(x :: AbstractMatrix)
    argmax.(eachcol(x))
end

params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)

#data = generate_xcat_ycat(params)
data = CSV.read("data.csv", DataFrame)
# -


onecold(onehot(data.Y)) ≈ data.Y

# +
function train!(model, x, y)
        
    loader = Flux.DataLoader((x, y), batchsize=1000, shuffle=true)
    optim = Flux.setup(Flux.Adam(0.1), model)
    
    for epoch in 1:500
        for (x, y) in loader
            grads = Flux.gradient(model) do m
                y_hat = m(x)
                Flux.binarycrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
    end
end

function simple_learning( data )
    
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    YB = dbb.Y
    ZA = dba.Z
    
    XA = onehot(Matrix(dba[!, [:X1, :X2, :X3]]))
    XB = onehot(Matrix(dbb[!, [:X1, :X2, :X3]]))
    
    YA = onehot(dba.Y)
    ZB = onehot(dbb.Z)
    
    dimX = size(XA, 1)
    dimY = size(YA, 1)
    dimZ = size(ZB, 1)
    
    nA = size(XA, 2)
    nB = size(XB, 2)
    
    modelXYA = Chain( Dense(dimX, 160, relu), Dense(160, dimY, relu), softmax)
    modelXZB = Chain( Dense(dimX, 160, relu), Dense(160, dimZ, relu), softmax)
    
    train!(modelXYA, XA, YA)
    train!(modelXZB, XB, ZB)
    
    YBpred = onecold(modelXYA(XB))
    ZApred = onecold(modelXZB(XA))
    
    @show sum(YB .== YBpred) / nA
    @show sum(ZA .== ZApred) / nB
    return (sum(YB .== YBpred) + sum(ZA .== ZApred)) / (nA + nB)
    
end
    
@time simple_learning( data )
# -





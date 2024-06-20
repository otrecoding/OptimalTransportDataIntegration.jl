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

# # Simple learning
# +
using OptimalTransportDataIntegration
using CSV
using DataFrames
import Flux
import Flux: Chain, Dense

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


# +
function train!(model, x, y; learning_rate = 0.01, batchsize=512, epochs = 500)
    
    loader = Flux.DataLoader((x, y), batchsize=batchsize, shuffle=true)
    optim = Flux.setup(Flux.Adam(learning_rate), model)
    
    for epoch in 1:epochs
        for (x, y) in loader
            grads = Flux.gradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
    end
        
end
# +
function simple_learning( data; hidden_layer_size = 10,  learning_rate = 0.01, batchsize=512, epochs = 500)

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    YB = dbb.Y
    ZA = dba.Z
    
    XA = onehot(Matrix(dba[!, [:X1, :X2, :X3]]))
    XB = onehot(Matrix(dbb[!, [:X1, :X2, :X3]]))
    
    YA = Flux.onehotbatch(dba.Y, 1:4)
    ZB = Flux.onehotbatch(dbb.Z, 1:3)
    
    dimX = size(XA, 1)
    dimY = size(YA, 1)
    dimZ = size(ZB, 1)
    
    nA = size(XA, 2)
    nB = size(XB, 2)
    
    modelXYA = Chain(Dense(dimX, hidden_layer_size),  Dense(hidden_layer_size, dimY))
    modelXZB = Chain(Dense(dimX, hidden_layer_size),  Dense(hidden_layer_size, dimZ))
    modelXYA = Chain(Dense(dimX, dimY))
    modelXZB = Chain(Dense(dimX, dimZ))
    
    la = train!(modelXYA, XA, YA)
    lb = train!(modelXZB, XB, ZB)
    
    YBpred = Flux.onecold(modelXYA(XB))
    ZApred = Flux.onecold(modelXZB(XA))
    
    (sum(YB .== YBpred) + sum(ZA .== ZApred)) / (nA + nB)
    
end

data = CSV.read(joinpath(@__DIR__, "data.csv"), DataFrame)


simple_learning( data, hidden_layer_size = i )




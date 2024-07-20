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
using Random
using Statistics, ProgressMeter
import Flux
using Plots

# Define the number of classes, samples, and features
num_classes = 4
num_samples = 1024
num_features = 2

# Seed for reproducibility
rng = MersenneTwister(42)

# Generate synthetic data
X = rand(rng,  num_features, num_samples,)  
Y = zeros(Int, num_samples) 
@. Y[ (X[1,:] <= 0.5) & (X[2,:] <= 0.5)] = 1
@. Y[ (X[1,:] > 0.5) & (X[2,:] <= 0.5)] = 2
@. Y[ (X[1,:] <= 0.5) & (X[2,:] > 0.5)] = 3
@. Y[ (X[1,:] > 0.5) & (X[2,:] > 0.5)] = 4
scatter(X[1,:], X[2,:], group = Y)
# +
import Flux: Chain, Dense, relu, softmax

function train!(model, loader, optim, epochs = 1000)
    
    losses = Float32[]
    loss = Inf
    @showprogress for epoch in 1:epochs
        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
        push!(losses, loss)
    end
end

function learning( X, Y, epochs = 1000, batchsize = 256)
    
    x = Float32.(X)
    y = Flux.onehotbatch(Y, 1:4)
    nx = size(x, 1)
    ny = size(y, 1)
    
    model = Chain( Dense(nx,  ny))
    loader = Flux.DataLoader((x, y), batchsize=batchsize, shuffle=true)
    optim = Flux.setup(Flux.Adam(0.01), model)
    train!(model, loader, optim )
    
    mean(Flux.onecold(model(x)) .== Y)
    
end
# -

learning( X, Y)

learning( X, Y)

learning( X, Y)

learning( X, Y)





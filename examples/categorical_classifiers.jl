# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Julia 1.9.3
#     language: julia
#     name: julia-1.9
# ---

# +
using Random
using Statistics, ProgressMeter
import Flux
using Plots

# Define the number of classes, samples, and features
num_classes = 4
num_samples = 1000
num_features = 2

# Seed for reproducibility
rng = MersenneTwister(42)

# Generate synthetic data
X = rand(rng,  num_features, num_samples,)  # Replace with your data generation logic
Y = zeros(Int, num_samples)  # Replace with your class assignment logic
@. Y[ (X[1,:] <= 0.5) & (X[2,:] <= 0.5)] = 1
@. Y[ (X[1,:] > 0.5) & (X[2,:] <= 0.5)] = 2
@. Y[ (X[1,:] <= 0.5) & (X[2,:] > 0.5)] = 3
@. Y[ (X[1,:] > 0.5) & (X[2,:] > 0.5)] = 4
scatter(X[1,:], X[2,:], group = Y)
# -
y = sort(unique(Y)) .== permutedims(Y)

# +
import Flux: Chain, Dense, relu, softmax

nx = size(X, 1)
ny = size(y, 1)

model = Chain( Dense(nx, 160, relu), Dense(160, ny, relu), softmax)
x = Float32.(X)
ŷ = model(x)
@show Flux.crossentropy(ŷ, y)

predict( model, x) = Flux.onecold(model(x))
accuracy(model, x, y) = mean(predict(model,x) .== y) 

# +
function train!(model, x, y, epochs = 1000, batchsize = 16)
    loader = Flux.DataLoader((x, y), batchsize=batchsize, shuffle=true)
    optim = Flux.setup(Flux.Adam(0.01), model)
    @showprogress for epoch in 1:epochs
        for (x, y) in loader
            grads = Flux.gradient(model) do m
                y_hat = m(x)
                Flux.binarycrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
    end
end

train!(model, x, y)

# + endofcell="--"
# -

# Looking at the accuracy

mean(Flux.onecold(model(x)) .== y)
# --



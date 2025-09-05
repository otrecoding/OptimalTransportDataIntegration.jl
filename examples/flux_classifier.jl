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

# # Flux classifier
#
# The following page contains a step-by-step walkthrough of a classifier implementation in Julia using Flux.
# Let's start by importing the required Julia packages.

using Statistics, DataFrames, CSV
using OptimalTransportDataIntegration
import Flux
import Flux: Chain, Dense, relu, softmax, onehotbatch, onecold, logitcrossentropy

# ## Dataset

params = DataParameters()
rng = DiscreteDataGenerator(params)
data = generate(rng)

dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

XA = OptimalTransportDataIntegration.onehot(Matrix(dba[!, [:X1, :X2, :X3]]))
XB = OptimalTransportDataIntegration.onehot(Matrix(dbb[!, [:X1, :X2, :X3]]))

YA = onehotbatch(dba.Y, 1:4)
ZB = onehotbatch(dbb.Z, 1:3)
# -


nx = size(XA, 1)
ny = size(YA, 1)
model1 = Chain(Dense(nx, ny))

nx = size(XB, 1)
ny = size(ZB, 1)
model2 = Chain(Dense(nx, ny))

# -

# ## Training the model

function train!(model, x, y, epochs = 1000, batchsize = 64)
    loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
    optim = Flux.setup(Flux.Adam(0.01), model)
    for epoch in 1:epochs
        for (x, y) in loader
            grads = Flux.gradient(model) do m
                y_hat = m(x)
                logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
    end
    return
end

train!(model1, XA, YA)
train!(model2, XB, ZB)

# -

# Looking at the accuracy

@show mean(Flux.onecold(model1(XB)) .== dbb.Y)
@show mean(Flux.onecold(model2(XA)) .== dba.Z)

method = SimpleLearning()

@show accuracy(otrecod(data, method))

rng = ContinuousDataGenerator(params)
data = generate(rng)

@show accuracy(otrecod(data, method))
@show accuracy(otrecod(data, JointOTBetweenBases()))

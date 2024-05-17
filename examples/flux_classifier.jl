# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Julia 1.9.3
#     language: julia
#     name: julia-1.9
# ---

# # Flux classifier
#
# The following page contains a step-by-step walkthrough of a classifier implementation in Julia using Flux. 
# Let's start by importing the required Julia packages.

using Statistics, DataFrames, CSV, ProgressMeter
using OptimalTransportDataIntegration
import Flux

# ## Dataset

# +
data = DataFrame(CSV.File("dataset.csv"))
data.X1_1 = data.X1
data.X2_1 =to_categorical(data.X2)[2,:]
data.X2_2 =to_categorical(data.X2)[3,:]
data.X3_1 =to_categorical(data.X3)[2,:]
data.X3_2 =to_categorical(data.X3)[3,:]
data.X3_3 =to_categorical(data.X3)[4,:]

dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

Xnames = [:X1_1,:X2_1,:X2_2, :X3_1, :X3_2, :X3_3]

XA = dba[!, Xnames]
XB = dbb[!, Xnames]

YA = to_categorical(dba.Y)
ZB = to_categorical(dbb.Z)
ZA = to_categorical(dba.Z)
YB = to_categorical(dbb.Y)
# -

# Our next step would be to convert this data into a form that can be fed to a machine learning model. The `x` values are arranged in a matrix and should ideally be converted to `Float32` type, but the labels must be one hot encoded.

x = Float32.(Matrix(XA)') 

y = YA

# A `Dense(3 => 160)` layer denotes a layer with 3 inputs (three features in every data point) and 4 outputs (four classes or labels). 
# The `softmax` function provided by NNLib.jl is re-exported by Flux, which has been used here. Lastly, Flux provides users with a `Chain` struct which makes stacking layers seamless.

# +
import Flux: Chain, Dense, relu, softmax

nx = size(x, 1)
ny = size(y, 1)

model = Chain( Dense(nx, 160, relu), Dense(160, ny, relu), softmax)
# -

# ## Loss and accuracy
#  
# Our next step should be to define some quantitative values for our model, which we will maximize or minimize during the complete training procedure. These values will be the loss function and the accuracy metric.
#  
#
# Flux provides us with many minimal yet elegant loss functions. The functions present in Flux includes sanity checks, ensures efficient performance, and behaves well with the overall FluxML ecosystem.

ŷ = model(x)
Flux.crossentropy(ŷ, y)

predict( model, x) = Flux.onecold(model(x))
accuracy(model, x, y) = mean(predict(model,x) .== y) 

# ## Training the model
#
# Let's train our model using the classic Gradient Descent algorithm. Here we will train the model for a maximum of `100` epochs.

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
# -

# Looking at the accuracy

mean(Flux.onecold(model(x)) .== dba.Y)



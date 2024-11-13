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
#     display_name: Julia 1.10.4
#     language: julia
#     name: julia-1.10
# ---

# +
using BetaML

# Creating test data..
X = rand(2000,2)
# note that the Y are 0.0/1.0 floats
Y = hcat(round.(tanh.(0.5 .* X[:,1] + 0.8 .* X[:,2])),
         round.(tanh.(0.5 .* X[:,1] + 0.3 .* X[:,2])),
         round.(tanh.(max.(0.0,-3 .* X[:,1].^2 + 2 * X[:,1] + 0.5 .* X[:,2]))))
# Creating the NN model...
l1 = DenseLayer(2,10,f = relu)
l2 = DenseLayer(10,3,f = x -> (tanh(x) + 1)/2)
mynn = NeuralNetworkEstimator(layers=[l1,l2],
      loss=squared_cost,
      descr="Multinomial logistic regression", 
    batch_size=8, epochs=100) 

res = fit!(mynn,X,Y) # Fit the model to the (scaled) data
# -

# Predictions...
ŷ = round.(predict(mynn,X))
(nrec,ncat) = size(Y) 
# Just a basic accuracy measure. I could think to extend the ConfusionMatrix measures to multi-label classification if needed..
overallAccuracy = sum(ŷ .== Y)/(nrec*ncat) # 0.999

# +
import Flux
import Statistics: mean

# Creating test data..
X = rand(2000, 2)
# note that the Y are 0.0/1.0 floats
Y = hcat(round.(tanh.(0.5 .* X[:,1] + 0.8 .* X[:,2])),
         round.(tanh.(0.5 .* X[:,1] + 0.3 .* X[:,2])),
         round.(tanh.(max.(0.0,-3 .* X[:,1].^2 + 2 * X[:,1] + 0.5 .* X[:,2]))))



# +
g = Flux.Chain( Flux.Dense(2, 10, Flux.relu), 
                    Flux.Dense(10, 3, x -> (tanh(x) + 1)/2))

loader = Flux.DataLoader((x, y), batchsize=8, shuffle=true)

optim = Flux.setup(Flux.Adam(0.01), model)

for epoch in 1:1_000
    for (x, y) in loader
        grads = Flux.gradient(model) do m
            y_hat = m(x)
            Flux.mse(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
    end
end
mean(round.(model(x)) .== y)
# -



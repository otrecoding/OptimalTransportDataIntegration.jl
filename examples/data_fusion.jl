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

using OptimalTransportDataIntegration
using Distributions
using DataFrames
import PythonOT
using Flux
using Distances
import LossFunctions

# +
params = DataParameters(nA = 1000, nB = 500)

data = generate_xcat_ycat(params)

# +
data.X1_1 = data.X1
data.X2_1 = Float32.(to_categorical(data.X2)[2,:])
data.X2_2 = Float32.(to_categorical(data.X2)[3,:])
data.X3_1 = Float32.(to_categorical(data.X3)[2,:])
data.X3_2 = Float32.(to_categorical(data.X3)[3,:])
data.X3_3 = Float32.(to_categorical(data.X3)[4,:])

dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

YBtrue = dbb.Y
ZAtrue = dba.Z

Xnames = [:X1_1,:X2_1,:X2_2, :X3_1, :X3_2, :X3_3]

# +
XA = Matrix{Float32}(dba[!, Xnames])
XB = Matrix{Float32}(dbb[!, Xnames])

YA = to_categorical(dba.Y)
ZB = to_categorical(dbb.Z)
ZA = to_categorical(dba.Z)
YB = to_categorical(dbb.Y)
nA = size(XA, 1)
nB = size(XB, 1)

wa = fill( 1 /nA, nA)
wb = fill( 1/nB, nB)

nbvarX = 3
C0 = pairwise(Hamming(), XA', XB'; dims = 2) ./ nbvarX
C = C0 ./ maximum(C0)

G = PythonOT.emd(wa, wb, C)
YB = nB .* G' * YA'
ZA = nA .* G * ZB'

XYA = hcat(XA, YA')
XZB = hcat(XB, ZB')

dimY = size(YA, 1)
dimZ = size(ZB, 1)
nA, dimXYA = size(XYA)
nB, dimXZB = size(XZB)

dimY, dimZ, dimXYA, dimXZB

# +
model = Chain(Dense(dimXYA =>160), Dense(160=>dimZ),  softmax)
# Define a custom hinge loss function
function hinge_loss(y_pred, y_true)
    return max(0.0f0, 1.0f0 .- y_pred .* y_true)
end

optim = Flux.setup(Flux.Adam(0.01), model)
X = permutedims(XYA)
y = permutedims(ZA)

loader = Flux.DataLoader((X, y), batchsize=32, shuffle=true)

optim = Flux.setup(Flux.Adam(0.01), model)

for epoch in 1:1_000
    for (X, y) in loader
        grads = Flux.gradient(model) do m
            y_hat = m(X)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
    end
end

yb = vec(getindex.(argmax(model(X), dims=1), 1))
# -
mean(ZAtrue .== yb)

XYA



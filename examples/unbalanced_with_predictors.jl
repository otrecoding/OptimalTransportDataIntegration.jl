# # OT method using predictors
#
# ## Data parameters
#
# The parameters we used to generate the dataset
using CSV
using DataFrames
using Distances
using JSON
using Flux
using OptimalTransportDataIntegration
using Test
import PythonOT

# +
json_file = joinpath("dataset.json")
csv_file = joinpath("dataset.csv")

const hidden_layer_size = 10
        
params = JSON.parsefile("dataset.json")

println(params)

# ## Dataset
#
# Read the csv file

data = CSV.read(csv_file, DataFrame)

T = Int32

# Split the two databases

base = data.database

indA = findall(base .== 1)
indB = findall(base .== 2)

X = OptimalTransportDataIntegration.onehot(Matrix(data[!, [:X1, :X2, :X3]]))
Y = Vector{T}(data.Y)
Z = Vector{T}(data.Z)

YBtrue = view(Y, indB)
ZAtrue = view(Z, indA)

YA = Flux.onehotbatch(Y[indA], 1:4)
ZB = Flux.onehotbatch(Z[indB], 1:3)

XA = view(X, :, indA)
XB = view(X, :, indB)

XYA = vcat(XA, YA)
XZB = vcat(XB, ZB)


nA :: Int = params["nA"]
nB :: Int = params["nB"]

@test nA == size(XA, 2)
@test nB == size(XB, 2)

wa = ones(nA) ./ nA
wb = ones(nB) ./ nB

C0 = pairwise(Hamming(), XA, XB)

C = C0 / maximum(C0)

dimXYA = size(XYA, 1)
dimXZB = size(XZB, 1)
dimYA = size(YA, 1)
dimZB = size(ZB, 1)

modelXYA = Chain(Dense(dimXYA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
modelXZB = Chain(Dense(dimXZB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

function train!(model, x, y; learning_rate = 0.01, batchsize = 64, epochs = 500)

    loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
    optim = Flux.setup(Flux.Adam(learning_rate), model)

    for epoch = 1:epochs
        for (x, y) in loader
            grads = Flux.gradient(model) do m
                y_hat = m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
        end
    end

end

wa = fill(1 / nA, nA)
wb = fill(1 / nB, nB)

G = PythonOT.emd(wa, wb, C)

#=
@show YB = nB .* YA * G
@show ZA = nA .* ZB * G'

train!(modelXYA, XYA, ZA)
train!(modelXZB, XZB, YB)

@show YB = modelXZB(XZB)
@show ZA = modelXYA(XYA)


g.fit(XZB, YB)
yBpred = g.predict(XZB)

f.fit(XYA, ZA)
zApred = f.predict(XYA)

YBpred = onecold(yBpred)
ZApred = onecold(zApred)  
est = (np.sum(YBtrue == YBpred) + np.sum(ZAtrue == ZApred)) / (nA + nB)
est
# -

# BCD algorithm, rerun the cell to check results

# +
numBCD = 10
alpha1, alpha2 = 0.25, 0.33

for i in range(numBCD):
    chinge1 = alpha1 * loss_crossentropy(yA, yBpred)
    chinge2 = alpha2 * loss_crossentropy(zB, zApred).T
    fcost = chinge1 + chinge2
    
    C = C0 / np.max(C0)  + fcost
    
    G = ot.emd(wa, wb, C)
    
    YB = nB * G.T.dot(YA)
    ZA = nA * G.dot(ZB)
    
    g.fit(XZB, YB)
    yBpred = g.predict(XZB)
    
    f.fit(XYA, ZA)
    zApred = f.predict(XYA)
    
    YBpred = onecold(yBpred)
    ZApred = onecold(zApred)  
    est = (np.sum(YBtrue == YBpred) + np.sum(ZAtrue == ZApred)) / (nA + nB)
    print(est)
# -


=#

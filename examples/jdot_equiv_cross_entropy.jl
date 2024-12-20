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

# +
json_file = joinpath("dataset.json")
csv_file = joinpath("dataset.csv")
        
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

# Compute distances matrix and save original vectors YA and ZB to yA and zB

distance = Hamming()

nA :: Int = params["nA"]
nB :: Int = params["nB"]

@test nA == size(XA, 2)
@test nB == size(XB, 2)

wa = ones(nA) ./ nA
wb = ones(nB) ./ nB

C0 = pairwise(distance, XA, XB) ./ size(XA, 1)

C = C0 / np.max(C0)

dimXA = size(XA, 1)
dimXB = size(XB, 1)
dimYA = size(YA, 1)
dimZB = size(ZB, 1)

modelXYA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
modelXZB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

train!(
        modelXYA,
        XA,
        YA,
        learning_rate = learning_rate,
        batchsize = batchsize,
        epochs = epochs,
)
train!(
        modelXZB,
        XB,
        ZB,
        learning_rate = learning_rate,
        batchsize = batchsize,
        epochs = epochs,
)

YB = Flux.onecold(modelXYA(XB))
ZA = Flux.onecold(modelXZB(XA))

#=



yA = np.copy(YA)
zB = np.copy(ZB)
# -


# ## Use a Neural Network for classifier

# +
from keras import initializers

def get_model_en(dimX, dimY):
    model = Sequential()
    model.add(Dense(160, input_dim=dimX, activation="relu"))
    model.add(Dense(dimY, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# -

# function to reset weights

# +
import tensorflow as tf

def reset_weights(model):
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))


# +
dimXZB = XZB.shape[1]
dimXYA = XYA.shape[1]
dimY = YA.shape[1]
dimZ = ZB.shape[1]
g = get_model_en(dimXZB, dimY)
f = get_model_en(dimXYA, dimZ)

C0 = cdist(XA, XB, metric="hamming") #* dimX / nbrvarX

C = C0 / np.max(C0)

G = ot.emd(wa, wb, C)

YB = nB * G.T.dot(yA)
ZA = nA * G.dot(zB)

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

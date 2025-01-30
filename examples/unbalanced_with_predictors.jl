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

function unbalanced_with_predictors(data; iterations = 10)

    T = Int32

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

    nA::Int = params["nA"]
    nB::Int = params["nB"]

    @assert nA == size(XA, 2)
    @assert nB == size(XB, 2)

    wa = ones(nA) ./ nA
    wb = ones(nB) ./ nB

    C0 = pairwise(Hamming(), XA, XB)

    C = C0 ./ maximum(C0)

    dimXYA = size(XYA, 1)
    dimXZB = size(XZB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    hidden_layer_size = 10
    modelXYA = Chain(Dense(dimXYA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXZB = Chain(Dense(dimXZB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

    function train!(model, x, y; learning_rate = 0.01, batchsize = 16, epochs = 500)

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


    alpha1, alpha2 = 0.25, 0.33

    function loss_crossentropy(Y, F)
        res = zeros(size(Y, 2), size(F, 2))
        logF = log.(F)
        for i in axes(Y, 1)
            res .+= -Y[i, :] .* logF[i, :]'
        end
        return res
    end

    for i = 1:iterations # BCD algorithm

        reg = 0.1
        G = PythonOT.entropic_partial_wasserstein(wa, wb, C, reg)

        YB = nB .* YA * G
        ZA = nA .* ZB * G'

        train!(modelXYA, XYA, ZA)
        train!(modelXZB, XZB, YB)

        YBpred = Flux.softmax(modelXZB(XZB))
        ZApred = Flux.softmax(modelXYA(XYA))

        est_y = sum(YBtrue .== Flux.onecold(YBpred)) / nB
        est_z = sum(ZAtrue .== Flux.onecold(ZApred)) / nA
        est =
            (sum(YBtrue .== Flux.onecold(YBpred)) + sum(ZAtrue .== Flux.onecold(ZApred))) /
            (nA + nB)

        println("est_y = $est_y ,  est_z = $est_z, est = $est")

        loss_y = alpha1 * loss_crossentropy(YA, YBpred)
        loss_z = alpha2 * loss_crossentropy(ZB, ZApred)

        fcost = loss_y .+ loss_z'

        println("fcost = $(sum(G .* fcost))")

        C .= C0 ./ maximum(C0) .+ fcost

        println("total cost = $(sum(G .* C))")

    end

end

json_file = joinpath("dataset.json")
csv_file = joinpath("dataset.csv")

params = JSON.parsefile("dataset.json")

data = CSV.read(csv_file, DataFrame)
@time unbalanced_with_predictors(data, iterations = 1)

@time println("OT : $(otrecod(data, OTjoint()))")
@time println("Simple Learning : $(otrecod(data, SimpleLearning()))")
@time println("OTE : $(otrecod(data, UnbalancedModality(iterations=0)))")

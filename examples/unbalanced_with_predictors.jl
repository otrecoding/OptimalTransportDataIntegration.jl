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

function unbalanced_with_predictors()

    # +
    json_file = joinpath("dataset.json")
    csv_file = joinpath("dataset.csv")
    
    hidden_layer_size = 10
            
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
    
    train!(modelXYA, XYA, ZA)
    train!(modelXZB, XZB, YB)
    
    YBpred_old = modelXZB(XZB)
    ZApred_old = modelXYA(XYA)
    
    @show est = (sum(YBtrue .== Flux.onecold(YBpred)) + sum(ZAtrue .== Flux.onecold(ZApred))) / (nA + nB)
    
    # BCD algorithm
    
    iterations = 10
    alpha1, alpha2 = 0.25, 0.33

    function loss_crossentropy(Y, F)
        ϵ = 1e-12
        res = zeros(size(Y, 2), size(F, 2))
        logF = log.(F .+ ϵ)
        for i in axes(Y, 1)
            res .+= -Y[i, :] .* logF[i, :]'
        end
        return res
    end

    for i in 1:iterations
     
        G = PythonOT.emd(wa, wb, C)
    
        YB = nB .* YA * G
        ZA = nA .* ZB * G'
    
        train!(modelXYA, XYA, ZA)
        train!(modelXZB, XZB, YB)
     
        YBpred_new = Flux.softmax(modelXZB(XZB))
        ZApred_new = Flux.softmax(modelXYA(XYA))
    
        loss_y = alpha1 * loss_crossentropy(YA, YBpred_new)
        loss_z = alpha2 * loss_crossentropy(ZB, ZApred_new)

        @show size(loss_y)
    
        fcost = loss_y .+ loss_z'
         
        C .= C0 / maximum(C0) .+ fcost
         
        @show est = (sum(YBtrue .== Flux.onecold(YBpred_new)) + sum(ZAtrue .== Flux.onecold(ZApred_new))) / (nA + nB)
     
    end

end

@time unbalanced_with_predictors()


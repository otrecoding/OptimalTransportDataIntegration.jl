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
import LinearAlgebra: norm
using ProgressMeter

function unbalanced_with_predictors(data; iterations = 10)

    T = Int32

    Ylevels = 1:4
    Zlevels = 1:3

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    XA = transpose(Matrix{Float32}(dba[!, [:X1, :X2, :X3]]))
    XB = transpose(Matrix{Float32}(dbb[!, [:X1, :X2, :X3]]))

    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)

    XYA = vcat(XA, YA)
    XZB = vcat(XB, ZB)

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(nA) ./ nA
    wb = ones(nB) ./ nB

    C0 = pairwise(Hamming(), XA, XB)

    C = C0 ./ maximum(C0)

    dimXYA = size(XYA, 1)
    dimXZB = size(XZB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    hidden_layer_size = 100
    modelXYA = Chain(Dense(dimXYA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXZB = Chain(Dense(dimXZB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

    function train!(model, x, y; learning_rate = 0.01, batchsize = 512, epochs = 500)

        loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
        optim = Flux.setup(Flux.Adam(learning_rate), model)

        @showprogress 1 for epoch = 1:epochs
            for (x, y) in loader
                grads = Flux.gradient(model) do m
                    y_hat = m(x)
                    Flux.logitcrossentropy(y_hat, y)
                end
                Flux.update!(optim, model, grads[1])
            end
        end

    end

    function loss_crossentropy(Y, F)

        ϵ = 1e-12
        res = zeros(size(Y, 2), size(F, 2))
        logF = similar(F)
        for i in eachindex(F)
            if F[i] ≈ 1.0
                logF[i] = log(1.0 - ϵ)
            else
                logF[i] = log(ϵ)
            end
        end

        for i in axes(Y, 1)
            res .+= -Y[i, :] .* logF[i, :]'
        end
        return res
    end

    YBpred = Flux.softmax(modelXZB(XZB))
    ZApred = Flux.softmax(modelXYA(XYA))

    alpha1, alpha2 = 0.25, 0.33

    G = ones(length(wa), length(wb))
    @show cost = Inf

    for iter = 1:iterations # BCD algorithm

        Gold = copy(G)
        costold = cost

        G = PythonOT.emd(wa, wb, C)

        @show delta = norm(G .- Gold)

        YB = nB .* YA * G
        ZA = nA .* ZB * G'

        @info "Train on base A"
        train!(modelXYA, XYA, ZA)
        @info "Train on base B"
        train!(modelXZB, XZB, YB)

        YBpred .= Flux.softmax(modelXZB(XZB))
        ZApred .= Flux.softmax(modelXYA(XYA))

        loss_y = alpha1 * loss_crossentropy(YA, YBpred)
        loss_z = alpha2 * loss_crossentropy(ZB, ZApred)

        fcost = loss_y .+ loss_z'

        cost = sum(G .* fcost)

        @info "Delta: $(delta) \t  Loss: $(cost) "

        if delta < 1e-16 || abs(costold - cost) < 1e-7
            @info "converged at iter $iter "
            break
        end

        C .= C0 ./ maximum(C0) .+ fcost

    end

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

rng = DataGenerator(DataParameters(), scenario = 1, discrete = false)

data = generate(rng)

@time yb, za = unbalanced_with_predictors(data, iterations = 10)

println(accuracy(data, yb, za))


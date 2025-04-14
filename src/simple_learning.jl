import Flux
import Flux: Chain, Dense

function onehot(x::AbstractMatrix)
    res = Vector{Float32}[]
    for col in eachcol(x)
        levels = filter(x -> x != 0, sort(unique(col)))
        for level in levels
            push!(res, col .== level)
        end
    end
    return stack(res, dims = 1)
end

function train!(model, x, y; learning_rate = 0.01, batchsize = 512, epochs = 500)

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

function simple_learning(
    data;
    hidden_layer_size = 10,
    learning_rate = 0.01,
    batchsize = 64,
    epochs = 1000,
)

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    XA = onehot(Matrix(dba[!, [:X1, :X2, :X3]]))
    XB = onehot(Matrix(dbb[!, [:X1, :X2, :X3]]))

    YA = Flux.onehotbatch(dba.Y, 1:4)
    ZB = Flux.onehotbatch(dbb.Z, 1:3)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    nA = size(XA, 2)
    nB = size(XB, 2)

    modelA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

    train!(
        modelA,
        XA,
        YA,
        learning_rate = learning_rate,
        batchsize = batchsize,
        epochs = epochs,
    )
    train!(
        modelB,
        XB,
        ZB,
        learning_rate = learning_rate,
        batchsize = batchsize,
        epochs = epochs,
    )

    YB = Flux.onecold(modelA(XB))
    ZA = Flux.onecold(modelB(XA))

    YB, ZA

end

import Flux
import Flux: Chain, Dense

function onehot(x::AbstractMatrix)
    res = Vector{Float32}[]
    for col in eachcol(x)
        levels = 1:4 #filter(x -> x != 0, sort(unique(col)))
        for level in levels
            push!(res, col .== level)
        end
    end
    return stack(res, dims = 1)
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

    colnames = names(data, r"^X")
    XA = onehot(Matrix(dba[!, colnames]))
    XB = onehot(Matrix(dbb[!, colnames]))

    YA = Flux.onehotbatch(dba.Y, 1:4)
    ZB = Flux.onehotbatch(dbb.Z, 1:3)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

    function train!(model, x, y)

        loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
        optim = Flux.setup(Flux.Adam(learning_rate), model)

        for epoch in 1:epochs
            for (x, y) in loader
                grads = Flux.gradient(model) do m
                    y_hat = m(x)
                    Flux.logitcrossentropy(y_hat, y)
                end
                Flux.update!(optim, model, grads[1])
            end
        end

        return
    end

    train!(modelA, XA, YA)
    train!(modelB, XB, ZB)

    YB = Flux.onecold(modelA(XB))
    ZA = Flux.onecold(modelB(XA))

    return YB, ZA

end

function joint_within_with_predictors(
        data;
        iterations = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 500,
        hidden_layer_size = 10,
        reg = 0.0,
        reg_m1 = 0.0,
        reg_m2 = 0.0
    )

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

    C0 = pairwise(Euclidean(), XA, XB, dims = 2)

    C = C0 ./ maximum(C0)

    dimXYA = size(XYA, 1)
    dimXZB = size(XZB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelXYA = Chain(Dense(dimXYA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXZB = Chain(Dense(dimXZB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

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

    if reg > 0
        G = PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G = PythonOT.emd(wa, wb, C)
    end

    delta = norm(G .- Gold)

    XB2 = nB .* XA * G
    XA2 = nA .* XB * G'

    YB = nB .* YA * G
    ZA = nA .* ZB * G'

    XYA = vcat(XA2, YA)
    XZB = vcat(XB2, ZB) 

    train!(modelXYA, XYA, ZA)
    train!(modelXZB, XZB, YB)

    YBpred .= modelXZB(XZB)
    ZApred .= modelXYA(XYA)

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

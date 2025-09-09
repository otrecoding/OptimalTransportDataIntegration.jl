function joint_between_without_yz(
        data;
        iterations = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 500,
        hidden_layer_size = 10,
        reg = 0.0,
        reg_m1 = 0.0,
        reg_m2 = 0.0,
        Ylevels = 1:4,
        Zlevels = 1:3
    )

    T = Int32

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    cols = names(dba, r"^X")   

    XA = transpose(Matrix{Float32}(dba[:, cols]))
    XB = transpose(Matrix{Float32}(dbb[:, cols]))

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

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)

    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelXA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

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

    function loss_crossentropy(Y, F)

        ϵ = 1.0e-12
        res = zeros(Float32, size(Y, 2), size(F, 2))
        logF = zeros(Float32, size(F))

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

    YBpred = Flux.softmax(modelXA(XB))
    ZApred = Flux.softmax(modelXB(XA))
    @show size(YBpred)
    @show size(ZApred)

    alpha1, alpha2 = 1 / length(Ylevels), 1 / length(Zlevels)

    G = ones(length(wa), length(wb))
    cost = Inf

    for iter in 1:iterations # BCD algorithm

        Gold = copy(G)
        costold = cost

        if reg > 0
            G = PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
        else
            G = PythonOT.emd(wa, wb, C)
        end

        delta = norm(G .- Gold)

        YB = nB .* YA * G
        ZA = nA .* ZB * G'

        train!(modelXA, XA, ZA)
        train!(modelXB, XB, YB)

        YBpred .= Flux.softmax(modelXA(XB))
        ZApred .= Flux.softmax(modelXB(XA))

        @show size(YBpred)
        @show size(ZApred)

        loss_y = alpha1 * loss_crossentropy(YA, YBpred)
        loss_z = alpha2 * loss_crossentropy(ZB, ZApred)

        fcost = loss_y .+ loss_z'

        cost = sum(G .* fcost)

        @info "Delta: $(delta) \t  Loss: $(cost) "

        if delta < 1.0e-16 || abs(costold - cost) < 1.0e-7
            @info "converged at iter $iter "
            break
        end

        C .= C0 ./ maximum(C0) .+ fcost

    end

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

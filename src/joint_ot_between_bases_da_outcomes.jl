function joint_ot_between_bases_da_outcomes(
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

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(nA) ./ nA
    wb = ones(nB) ./ nB

    C = Float32.(pairwise(SqEuclidean(), XA, XB, dims = 2))
    G = ones(Float32, length(wa), length(wb))

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end
    row_sumsB = vec(sum(G, dims=2))
    col_sumsA = vec(sum(G, dims=1))
    ZApred = Flux.softmax( vec(ZB * G').* (1 ./row_sumsB)')
    YBpred = Flux.softmax( vec(YA * G).* (1 ./col_sumsA)')


    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

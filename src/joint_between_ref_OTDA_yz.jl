function joint_between_ref_otda_yz(
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

    cols = names(dba, r"^X")   

    XA = transpose(Matrix(dba[:, cols]))
    XB = transpose(Matrix(dbb[:, cols]))

    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(nA) ./ nA
    wb = ones(nB) ./ nB

    C0 = pairwise(Euclidean(), XA, XB, dims = 2)

    C = C0 ./ maximum(C0)

    G = ones(length(wa), length(wb))

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end

    ZApred = nA .* ZB * G'
    YBpred = nB .* YA * G

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

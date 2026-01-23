export generate_mixed_data

function generate_mixed_data(params)

    q = length(params.mA)

    XDA = stack([rand(Categorical(params.pA[i]), params.nA) for i in 1:q], dims = 1)
    XDB = stack([rand(Categorical(params.pB[i]), params.nB) for i in 1:q], dims = 1)
    XCA = rand(MvNormal(params.mA, params.covA), params.nA)
    XCB = rand(MvNormal(params.mB, params.covB), params.nB)

    XA = vcat(XDA, XCA)
    XB = vcat(XDA, XCA)

    aA = vcat(params.aA, params.aA)
    aB = vcat(params.aB, params.aB)

    Y1 = XA' * aA
    Y2 = XB' * aB

    bYA = quantile(Y1, [0.25, 0.5, 0.75])
    bYB = quantile(Y2, [0.25, 0.5, 0.75])
    bZA = quantile(Y1, [1 / 3, 2 / 3])
    bZB = quantile(Y2, [1 / 3, 2 / 3])

    binsYA = vcat(-Inf, bYA, Inf)
    binsZA = vcat(-Inf, bZA, Inf)
    binsYB = vcat(-Inf, bYB, Inf)
    binsZB = vcat(-Inf, bZB, Inf)

    YA = digitize(Y1, binsYA)
    ZA = digitize(Y1, binsZA)

    YB = digitize(Y2, binsYB)
    ZB = digitize(Y2, binsZB)

    colnames = Symbol.("X" .* string.(1:2q))

    df = DataFrame(hcat(XA, XB)', colnames)

    df.Y = categorical(vcat(YA, fill(missing, length(YB))))
    df.Z = categorical(vcat(fill(missing, length(ZA)), ZB))
    df.Ytrue = categorical(vcat(YA, YB))
    df.Ztrue = categorical(vcat(ZA, ZB))
    df.database = categorical(vcat(fill(1, params.nA), fill(2, params.nB)))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))" #
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

export generate_real_discrete_data

function generate_real_discrete_data(params)

    q = length(params.mA)

    XA = stack([rand(Categorical(params.pA[i]), params.nA) for i in 1:q], dims = 1)
    XB = stack([rand(Categorical(params.pB[i]), params.nB) for i in 1:q], dims = 1)

    aA = params.aA
    aB = params.aB

    Y1 = XA' * aA
    Y2 = XB' * aB

    bYA = quantile(Y1, [0.25, 0.5, 0.75])
    bYB = quantile(Y2, [0.25, 0.5, 0.75])
    bZA = quantile(Y1, [1 / 3, 2 / 3])
    bZB = quantile(Y2, [1 / 3, 2 / 3])

    binsYA = vcat(-Inf, bYA, Inf)
    binsZA = vcat(-Inf, bZA, Inf)
    binsYB = vcat(-Inf, bYB, Inf)
    binsZB = vcat(-Inf, bZB, Inf)

    YA = digitize(Y1, binsYA)
    ZA = digitize(Y1, binsZA)

    YB = digitize(Y2, binsYB)
    ZB = digitize(Y2, binsZB)

    colnames = Symbol.("X" .* string.(1:q))

    df = DataFrame(hcat(XA, XB)', colnames)

    df.Y = categorical(vcat(YA, fill(missing, length(YB))), levels = 1:4)
    df.Z = categorical(vcat(fill(missing, length(ZA)), ZB), levels = 1:3)
    df.Ytrue = categorical(vcat(YA, YB), levels = 1:4)
    df.Ztrue = categorical(vcat(ZA, ZB), levels = 1:3)
    df.database = categorical(vcat(fill(1, params.nA), fill(2, params.nB)))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))" #
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

export generate_real_continuous_data

function generate_real_continuous_data(params)

    q = length(params.mA)

    XA = rand(MvNormal(params.mA, params.covA), params.nA)
    XB = rand(MvNormal(params.mB, params.covB), params.nB)

    aA = params.aA
    aB = params.aB

    Y1 = XA' * aA
    Y2 = XB' * aB

    bYA = quantile(Y1, [0.25, 0.5, 0.75])
    bYB = quantile(Y2, [0.25, 0.5, 0.75])
    bZA = quantile(Y1, [1 / 3, 2 / 3])
    bZB = quantile(Y2, [1 / 3, 2 / 3])

    binsYA = vcat(-Inf, bYA, Inf)
    binsZA = vcat(-Inf, bZA, Inf)
    binsYB = vcat(-Inf, bYB, Inf)
    binsZB = vcat(-Inf, bZB, Inf)

    YA = digitize(Y1, binsYA)
    ZA = digitize(Y1, binsZA)

    YB = digitize(Y2, binsYB)
    ZB = digitize(Y2, binsZB)

    colnames = Symbol.("X" .* string.(1:q))

    X = hcat(XA, XB)'
    df = DataFrame(X, colnames)

    df.Y = categorical(vcat(YA, fill(missing, length(YB))))
    df.Z = categorical(vcat(fill(missing, length(ZA)), ZB))
    df.Ytrue = categorical(vcat(YA, YB))
    df.Ztrue = categorical(vcat(ZA, ZB))
    df.database = categorical(vcat(fill(1, params.nA), fill(2, params.nB)))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))" #
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end


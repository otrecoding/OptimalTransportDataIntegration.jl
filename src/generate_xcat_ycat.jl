using Distributions
using DataFrames

export generate_xcat_ycat

function generate_XY(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3)

    dA = MvNormal(params.mA, params.covA)
    dB = MvNormal(params.mB, params.covB)

    X_glob1 = rand(dA, params.nA)
    X_glob2 = rand(dB, params.nB)

    px1cc = cumsum(params.px1c)[1:end-1]
    px2cc = cumsum(params.px2c)[1:end-1]
    px3cc = cumsum(params.px3c)[1:end-1]

    qxA1c = quantile.(Normal(params.mA[1], 1.0), px1cc)
    qxA2c = quantile.(Normal(params.mA[2], 1.0), px2cc)
    qxA3c = quantile.(Normal(params.mA[3], 1.0), px3cc)

    qxB1c = quantile.(Normal(params.mB[1], 1.0), px1cc)
    qxB2c = quantile.(Normal(params.mB[2], 1.0), px2cc)
    qxB3c = quantile.(Normal(params.mB[3], 1.0), px3cc)


    binsA1 = vcat(minimum(X_glob1[1, :]) - 100, qxA1c, maximum(X_glob1[1, :]) + 100)
    binsA2 = vcat(minimum(X_glob1[2, :]) - 100, qxA2c, maximum(X_glob1[2, :]) + 100)
    binsA3 = vcat(minimum(X_glob1[3, :]) - 100, qxA3c, maximum(X_glob1[3, :]) + 100)

    binsB1 = vcat(minimum(X_glob2[1, :]) - 100, qxB1c, maximum(X_glob2[1, :]) + 100)
    binsB2 = vcat(minimum(X_glob2[2, :]) - 100, qxB2c, maximum(X_glob2[2, :]) + 100)
    binsB3 = vcat(minimum(X_glob2[3, :]) - 100, qxB3c, maximum(X_glob2[3, :]) + 100)


    X11 = digitize(X_glob1[1, :], binsA1)
    X12 = digitize(X_glob1[2, :], binsA2)
    X13 = digitize(X_glob1[3, :], binsA3)

    X21 = digitize(X_glob2[1, :], binsB1)
    X22 = digitize(X_glob2[2, :], binsB2)
    X23 = digitize(X_glob2[3, :], binsB3)

    X11c = to_categorical(X11, 1:2)[2:end, :]
    X21c = to_categorical(X21, 1:2)[2:end, :]
    X12c = to_categorical(X12, 1:3)[2:end, :]
    X22c = to_categorical(X22, 1:3)[2:end, :]
    X13c = to_categorical(X13, 1:4)[2:end, :]
    X23c = to_categorical(X23, 1:4)[2:end, :]

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)
    R2 = params.r2

    dimU = 3

	varerrorA =
        (1 / R2 - 1) * sum([params.aA[i] * params.aA[j] * params.covA[i, j] for i = 1:dimU, j = 1:dimU])
  	varerrorB =
        (1 / R2 - 1) * sum([params.aB[i] * params.aB[j] * params.covB[i, j] for i = 1:dimU, j = 1:dimU])

   	Y1 = vcat(X11c, X12c, X13c)' * params.aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)

    Y2 = vcat(X21c, X22c, X23c)' * params.aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)

    X1, X2, X3, Y1, Y2

end

function generate_dataframe(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3, binsYA1, binsYB1, binsYA2, binsYB2)

    X1, X2, X3, Y1, Y2 = generate_XY(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3)

    YA1 = digitize(Y1, binsYA1)
    YA2 = digitize(Y1, binsYA2)

    binsYB1eps = binsYB1 .+ params.eps
    binsYB2eps = binsYB2 .+ params.eps

    YB1 = digitize(Y2, binsYB1eps)
    YB2 = digitize(Y2, binsYB2eps)

    df = DataFrame(hcat(X1, X2, X3) .- 1, [:X1, :X2, :X3])
    df.Y = vcat(YA1, YB1)
    df.Z = vcat(YA2, YB2)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    return df

end


"""
$(SIGNATURES)

Function to generate data where X and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

"""
function generate_xcat_ycat(params::DataParameters)

    dA = MvNormal(params.mA, params.covA)
    XA = rand(dA, params.nA)
    dB = MvNormal(params.mB, params.covB)
    XB = rand(dB, params.nB)

    px1cc = cumsum(params.px1c)[1:end-1]
    px2cc = cumsum(params.px2c)[1:end-1]
    px3cc = cumsum(params.px3c)[1:end-1]

    qxA1c = quantile.(Normal(params.mA[1], 1.0), px1cc)
    qxA2c = quantile.(Normal(params.mA[2], 1.0), px2cc)
    qxA3c = quantile.(Normal(params.mA[3], 1.0), px3cc)

    qxB1c = quantile.(Normal(params.mB[1], 1.0), px1cc)
    qxB2c = quantile.(Normal(params.mB[2], 1.0), px2cc)
    qxB3c = quantile.(Normal(params.mB[3], 1.0), px3cc)

    binsA1 = vcat(minimum(XA[1, :]) - 100, qxA1c, maximum(XA[1, :]) + 100)
    binsA2 = vcat(minimum(XA[2, :]) - 100, qxA2c, maximum(XA[2, :]) + 100)
    binsA3 = vcat(minimum(XA[3, :]) - 100, qxA3c, maximum(XA[3, :]) + 100)

    binsB1 = vcat(minimum(XB[1, :]) - 100, qxB1c, maximum(XB[1, :]) + 100)
    binsB2 = vcat(minimum(XB[2, :]) - 100, qxB2c, maximum(XB[2, :]) + 100)
    binsB3 = vcat(minimum(XB[3, :]) - 100, qxB3c, maximum(XB[3, :]) + 100)

    X1, X2, X3, Y1, Y2 = generate_XY(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3)

    bA1 = quantile(Y1, [0.25, 0.5, 0.75])
    bB1 = quantile(Y2, [0.25, 0.5, 0.75])
    bA2 = quantile(Y1, [1 / 3, 2 / 3])
    bB2 = quantile(Y2, [1 / 3, 2 / 3])

    binsYA1 = vcat(minimum(Y1) - 100, bA1, maximum(Y1) + 100)
    binsYB1 = vcat(minimum(Y1) - 100, bB1, maximum(Y1) + 100)

    binsYA2 = vcat(minimum(Y2) - 100, bA2, maximum(Y2) + 100)
    binsYB2 = vcat(minimum(Y2) - 100, bB2, maximum(Y2) + 100)


    df = generate_dataframe(params, binsA1, binsA2, binsA3, binsB1, binsB2, 
                            binsB3, binsYA1, binsYB1, binsYA2, binsYB2)

    @info "Categories in Y $(sort(unique(df.Y)))"
    @info "Catgeories in Z $(sort(unique(df.Z)))"

    return df

end

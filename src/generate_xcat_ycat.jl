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

    X11c = to_categorical(X11)[2:end, :]
    X21c = to_categorical(X21)[2:end, :]
    X12c = to_categorical(X12)[2:end, :]
    X22c = to_categorical(X22)[2:end, :]
    X13c = to_categorical(X13)[2:end, :]
    X23c = to_categorical(X23)[2:end, :]

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)

    Y1 = vcat(X11c, X12c, X13c)' * params.aA
    Y2 = vcat(X21c, X22c, X23c)' * params.aB

    X1, X2, X3, Y1, Y2
#####ou X11, X21,X12, X22,X13, X23
end

function generate_YZ(params, Y1, Y2)

    p = params.p

    py = [
        1 - p,
        (p - p^2) / 2,
        (p - p^2) / 2,
        (p^2 - p^3) / 2,
        (p^2 - p^3) / 2,
        p^3 / 4,
        p^3 / 4,
        p^3 / 4,
        p^3 / 4,
    ]

    Y = collect(0:(params.nA-1))
    Z = collect(0:(params.nB-1))

    U = rand(Multinomial(1, py), params.nA)
    V = rand(Multinomial(1, py), params.nB)

    UU = vec(sum(U .* collect(0:8), dims = 1))
    VV = vec(sum(V .* collect(0:8), dims = 1))

    Y[UU.==0] .= Y1[UU.==0]
    Y[UU.==1] .= Y1[UU.==1] .- 1
    Y[UU.==2] .= Y1[UU.==2] .+ 1
    Y[UU.==3] .= Y1[UU.==3] .+ 2
    Y[UU.==4] .= Y1[UU.==4] .- 2
    Y[UU.==5] .= 0
    Y[UU.==6] .= 1
    Y[UU.==7] .= 2
    Y[UU.==8] .= 3
    Y[Y.>3] .= 3
    Y[Y.<0] .= 0

    Z[VV.==0] .= Y2[VV.==0]
    Z[VV.==1] .= Y2[VV.==1] .- 1
    Z[VV.==2] .= Y2[VV.==2] .+ 1
    Z[VV.==3] .= Y2[VV.==3] .+ 2
    Z[VV.==4] .= Y2[VV.==4] .- 2
    Z[VV.==5] .= 0
    Z[VV.==6] .= 1
    Z[VV.==7] .= 2
    Z[VV.==8] .= 3
    Z[Z.>3] .= 3
    Z[Z.<0] .= 0

    return Y, Z

end

function generate_dataframe(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3, binsYA1, binsYB1, binsYA2, binsYB2)

    X1, X2, X3, Y1, Y2 = generate_XY(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3)
    Y, Z = generate_YZ(params, Y1, Y2)

    YA1 = digitize(Y, binsYA1)
    YA2 = digitize(Y, binsYA2)

    binsYB1eps = binsYB1 .+ params.eps
    binsYB2eps = binsYB2 .+ params.eps

    YB1 = digitize(Z, binsYB1eps)
    YB2 = digitize(Z, binsYB2eps)

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

    Y, Z = generate_YZ(params, Y1, Y2)
    bA1 = quantile(Y, [0.25, 0.5, 0.75])
    bB1 = quantile(Z, [0.25, 0.5, 0.75])
    bA2 = quantile(Y, [1 / 3, 2 / 3])
    bB2 = quantile(Z, [1 / 3, 2 / 3])


    binsYA1 = vcat(minimum(Y) - 100, bA1, maximum(Y) + 100)
    binsYB1 = vcat(minimum(Y) - 100, bB1, maximum(Y) + 100)

    binsYA2 = vcat(minimum(Z) - 100, bA2, maximum(Z) + 100)
    binsYB2 = vcat(minimum(Z) - 100, bB2, maximum(Z) + 100)


    df = generate_dataframe(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3, binsYA1, binsYB1, binsYA2, binsYB2)

    my = length(unique(df.Y))
    if my < 4
        @warn "Number of modality in Y $my < 4"
    end
    mz = length(unique(df.Z))
    if mz < 3
        @warn "Number of modality in Z $mz < 3"
    end

    return df

end

using Distributions
using DataFrames

export PDataGenerator

digitize(x, bins) = searchsortedlast.(Ref(bins), x)

to_categorical(x) = sort(unique(x)) .== permutedims(x)

to_categorical(x, levels) = levels .== permutedims(x)

struct PDataGenerator

    params :: DataParameters
    bins11 :: Vector{Float64}
    bins12 :: Vector{Float64}
    bins13 :: Vector{Float64}
    binsY11 :: Vector{Float64}
    binsY22 :: Vector{Float64}

    function PDataGenerator(params, n = 10000)

        dA = MvNormal(params.mA, params.covA)
        dB = MvNormal(params.mB, params.covB)

        XA = rand(dA, n)
        XB = rand(dB, n)

        px1cc = cumsum(params.px1c)[1:end-1]
        px2cc = cumsum(params.px2c)[1:end-1]
        px3cc = cumsum(params.px3c)[1:end-1]

        qx1c = quantile.(Normal(0.0, 1.0), px1cc)
        qx2c = quantile.(Normal(0.0, 1.0), px2cc)
        qx3c = quantile.(Normal(0.0, 1.0), px3cc)

        bins11 = vcat(minimum(XA[1, :]) - 100, qx1c, maximum(XA[1, :]) + 100)
        bins12 = vcat(minimum(XA[2, :]) - 100, qx2c, maximum(XA[2, :]) + 100)
        bins13 = vcat(minimum(XA[3, :]) - 100, qx3c, maximum(XA[3, :]) + 100)

        X11 = digitize(XA[1, :], bins11)
        X12 = digitize(XA[2, :], bins12)
        X13 = digitize(XA[3, :], bins13)

        X21 = digitize(XB[1, :], bins11)
        X22 = digitize(XB[2, :], bins12)
        X23 = digitize(XB[3, :], bins13)

        X11c = to_categorical(X11, 1:2)[2:end, :]
        X21c = to_categorical(X21, 1:2)[2:end, :]
        X12c = to_categorical(X12, 1:3)[2:end, :]
        X22c = to_categorical(X22, 1:3)[2:end, :]
        X13c = to_categorical(X13, 1:4)[2:end, :]
        X23c = to_categorical(X23, 1:4)[2:end, :]

        X1 = vcat(X11, X21)
        X2 = vcat(X12, X22)
        X3 = vcat(X13, X23)

        Y1 = vcat(X11c, X12c, X13c)' * params.aA
        Y2 = vcat(X21c, X22c, X23c)' * params.aB

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

        Y = Float64.(collect(0:(n-1)))
        Z = Float64.(collect(0:(n-1)))

        U = rand(Multinomial(1, py), n)
        V = rand(Multinomial(1, py), n)

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

        b11 = quantile(Y, [0.25, 0.5, 0.75])
        b22 = quantile(Z, [1 / 3, 2 / 3])

        binsY11 = vcat(minimum(Y) - 100, b11, maximum(Y) + 100)
        binsY22 = vcat(minimum(Z) - 100, b22, maximum(Z) + 100)

        new(params, bins11, bins12, bins13, binsY11, binsY22)

    end

end



"""
$(SIGNATURES)

Function to generate data where X and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

"""
function generate_data(generator::PDataGenerator)

    params = generator.params

    dA = MvNormal(params.mA, params.covA)
    dB = MvNormal(params.mB, params.covB)

    XA = rand(dA, params.nA)
    XB = rand(dB, params.nB)

    X11 = digitize(XA[1, :], generator.bins11)
    X12 = digitize(XA[2, :], generator.bins12)
    X13 = digitize(XA[3, :], generator.bins13)

    X21 = digitize(XB[1, :], generator.bins11)
    X22 = digitize(XB[2, :], generator.bins12)
    X23 = digitize(XB[3, :], generator.bins13)

    X11c = to_categorical(X11, 1:2)[2:end, :]
    X21c = to_categorical(X21, 1:2)[2:end, :]
    X12c = to_categorical(X12, 1:3)[2:end, :]
    X22c = to_categorical(X22, 1:3)[2:end, :]
    X13c = to_categorical(X13, 1:4)[2:end, :]
    X23c = to_categorical(X23, 1:4)[2:end, :]

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)

    Y1 = vcat(X11c, X12c, X13c)' * params.aA
    Y2 = vcat(X21c, X22c, X23c)' * params.aB

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

    Y = Float64.(collect(0:(params.nA-1)))
    Z = Float64.(collect(0:(params.nB-1)))

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

    Y11 = digitize(Y, generator.binsY11)
    Y12 = digitize(Y, generator.binsY22)

    binsY11eps = generator.binsY11 .+ params.eps
    binsY22eps = generator.binsY22 .+ params.eps

    Y21 = digitize(Z, binsY11eps)
    Y22 = digitize(Z, binsY22eps)

    df = DataFrame(hcat(X1, X2, X3) .- 1, [:X1, :X2, :X3])
    df.Y = vcat(Y11, Y21)
    df.Z = vcat(Y12, Y22)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in Y $(sort(unique(df.Y))) "
    @info "Categories in Z $(sort(unique(df.Z))) "

    return df

end

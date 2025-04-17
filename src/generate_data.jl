using Distributions
using DataFrames

export DataGenerator

export generate_data

struct DataGenerator

    params::DataParameters
    binsA1::Vector{Float64}
    binsA2::Vector{Float64}
    binsA3::Vector{Float64}
    binsB1::Vector{Float64}
    binsB2::Vector{Float64}
    binsB3::Vector{Float64}
    covAemp::Matrix{Float64}
    covBemp::Matrix{Float64}
    binsYA1::Vector{Float64}
    binsYB1::Vector{Float64} 
    binsYA2::Vector{Float64}
    binsYB2::Vector{Float64}

    function DataGenerator(params; n = 10000)

        dA = MvNormal(params.mA, params.covA)
        dB = MvNormal(params.mB, params.covB)

        XA = rand(dA, n)
        XB = rand(dB, n)

        px1cc = cumsum(params.px1c)[1:end-1]
        px2cc = cumsum(params.px2c)[1:end-1]
        px3cc = cumsum(params.px3c)[1:end-1]

        qxA1c = quantile.(Normal(params.mA[1], sqrt(params.covA[1,1])), px1cc)
        qxA2c = quantile.(Normal(params.mA[2], sqrt(params.covA[2,2])), px2cc)
        qxA3c = quantile.(Normal(params.mA[3], sqrt(params.covA[3,3])), px3cc)

        qxB1c = quantile.(Normal(params.mB[1], sqrt(params.covA[1,1])), px1cc)
        qxB2c = quantile.(Normal(params.mB[2], sqrt(params.covA[2,2])), px2cc)
        qxB3c = quantile.(Normal(params.mB[3], sqrt(params.covA[3,3])), px3cc)

        binsA1 = vcat(minimum(XA[1, :]) - 100, qxA1c, maximum(XA[1, :]) + 100)
        binsA2 = vcat(minimum(XA[2, :]) - 100, qxA2c, maximum(XA[2, :]) + 100)
        binsA3 = vcat(minimum(XA[3, :]) - 100, qxA3c, maximum(XA[3, :]) + 100)

        binsB1 = vcat(minimum(XB[1, :]) - 100, qxB1c, maximum(XB[1, :]) + 100)
        binsB2 = vcat(minimum(XB[2, :]) - 100, qxB2c, maximum(XB[2, :]) + 100)
        binsB3 = vcat(minimum(XB[3, :]) - 100, qxB3c, maximum(XB[3, :]) + 100)

        X11 = digitize(XA[1, :], binsA1)
        X12 = digitize(XA[2, :], binsA2)
        X13 = digitize(XA[3, :], binsA3)

        X21 = digitize(XB[1, :], binsB1)
        X22 = digitize(XB[2, :], binsB2)
        X23 = digitize(XB[3, :], binsB3)

        X11c = to_categorical(X11, 1:2)[2:end, :]
        X21c = to_categorical(X21, 1:2)[2:end, :]
        X12c = to_categorical(X12, 1:3)[2:end, :]
        X22c = to_categorical(X22, 1:3)[2:end, :]
        X13c = to_categorical(X13, 1:4)[2:end, :]
        X23c = to_categorical(X23, 1:4)[2:end, :]

   	    X1 = vcat(X11c, X12c, X13c)
        X2 = vcat(X21c, X22c, X23c)

        covAemp = cov(X1, dims=2)
        covBemp = cov(X2, dims=2)

        cr2 = 1 / params.r2 - 1

        aA = params.aA
        aB = params.aB

        covA = params.covA
        covB = params.covB

        varerrorA = cr2 * sum([aA[i] * aA[j] * covA[i, j] for i = axes(covA,1), j = axes(covA,2)])
        varerrorB = cr2 * sum([aB[i] * aB[j] * covB[i, j] for i = axes(covB,1), j = axes(covB,2)])

   	    Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), n)
        Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), n)

        bA1 = quantile(Y1, [0.25, 0.5, 0.75])
        bB1 = quantile(Y2, [0.25, 0.5, 0.75])
        bA2 = quantile(Y1, [1 / 3, 2 / 3])
        bB2 = quantile(Y2, [1 / 3, 2 / 3])

        binsYA1 = vcat(minimum(Y1) - 100, bA1, maximum(Y1) + 100)
        binsYB1 = vcat(minimum(Y1) - 100, bB1, maximum(Y1) + 100)

        binsYA2 = vcat(minimum(Y2) - 100, bA2, maximum(Y2) + 100)
        binsYB2 = vcat(minimum(Y2) - 100, bB2, maximum(Y2) + 100)

        new(params, binsA1, binsA2, binsA3, binsB1, binsB2, binsB3, covAemp, covBemp, 
            binsYA1, binsYB1, binsYA2, binsYB2) 

    end

end


"""
$(SIGNATURES)

Function to generate data where X and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

r2 is the coefficient of determination 

"""
function generate_data(generator::DataGenerator)

    params = generator.params

    dA = MvNormal(params.mA, params.covA)
    XA = rand(dA, params.nA)
    dB = MvNormal(params.mB, params.covB)
    XB = rand(dB, params.nB)

    X11 = digitize(XA[1, :], generator.binsA1)
    X12 = digitize(XA[2, :], generator.binsA2)
    X13 = digitize(XA[3, :], generator.binsA3)

    X21 = digitize(XB[1, :], generator.binsB1)
    X22 = digitize(XB[2, :], generator.binsB2)
    X23 = digitize(XB[3, :], generator.binsB3)

    X11c = to_categorical(X11, 1:2)[2:end, :]
    X21c = to_categorical(X21, 1:2)[2:end, :]
    X12c = to_categorical(X12, 1:3)[2:end, :]
    X22c = to_categorical(X22, 1:3)[2:end, :]
    X13c = to_categorical(X13, 1:4)[2:end, :]
    X23c = to_categorical(X23, 1:4)[2:end, :]

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)

    cr2 = 1. / params.r2 - 1

    aA = params.aA
    aB = params.aB

    covA = generator.covAemp
    covB = generator.covBemp

    varerrorA = cr2 * sum([aA[i] * aA[j] * covA[i, j] for i = axes(covA,1), j = axes(covA,2)])
    varerrorB = cr2 * sum([aB[i] * aB[j] * covB[i, j] for i = axes(covB,1), j = axes(covB,2)])

   	Y1 = vcat(X11c, X12c, X13c)' * params.aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
    Y2 = vcat(X21c, X22c, X23c)' * params.aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)

    YA1 = digitize(Y1, generator.binsYA1)
    YA2 = digitize(Y1, generator.binsYA2)

    YB1 = digitize(Y2, generator.binsYB1 )
    YB2 = digitize(Y2, generator.binsYB2 )

    df = DataFrame(hcat(X1, X2, X3) .- 1, [:X1, :X2, :X3])
    df.Y = vcat(YA1, YB1)
    df.Z = vcat(YA2, YB2)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in Y $(sort(unique(df.Y)))"
    @info "Categories in Z $(sort(unique(df.Z)))"

    return df

end

using Distributions
using DataFrames
import StatsBase: countmap
import OrderedCollections: OrderedDict

digitize(x, bins) = searchsortedlast.(Ref(bins), x)

to_categorical(x) = sort(unique(x)) .== permutedims(x)

to_categorical(x, levels) = levels .== permutedims(x)


export DataGenerator

export generate

struct DataGenerator

    params::DataParameters
    binsA1::Vector{Float64}
    binsA2::Vector{Float64}
    binsA3::Vector{Float64}
    covAemp::Matrix{Float64}
    covBemp::Matrix{Float64}
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}
    discrete::Bool

    function DataGenerator(params; n = 10000, scenario = 2, discrete = true)

        dA = MvNormal(params.mA, params.covA)
        dB = MvNormal(params.mB, params.covB)

        XA = rand(dA, n)
        XB = rand(dB, n)

        px1cc = cumsum(params.px1c)[1:(end - 1)]
        px2cc = cumsum(params.px2c)[1:(end - 1)]
        px3cc = cumsum(params.px3c)[1:(end - 1)]

        qxA1c = quantile.(Normal(params.mA[1], sqrt(params.covA[1, 1])), px1cc)
        qxA2c = quantile.(Normal(params.mA[2], sqrt(params.covA[2, 2])), px2cc)
        qxA3c = quantile.(Normal(params.mA[3], sqrt(params.covA[3, 3])), px3cc)

        qxB1c = quantile.(Normal(params.mB[1], sqrt(params.covA[1, 1])), px1cc)
        qxB2c = quantile.(Normal(params.mB[2], sqrt(params.covA[2, 2])), px2cc)
        qxB3c = quantile.(Normal(params.mB[3], sqrt(params.covA[3, 3])), px3cc)

        binsA1 = vcat(-Inf, qxA1c, Inf)
        binsA2 = vcat(-Inf, qxA2c, Inf)
        binsA3 = vcat(-Inf, qxA3c, Inf)

        if discrete

            X11 = digitize(XA[1, :], binsA1)
            X12 = digitize(XA[2, :], binsA2)
            X13 = digitize(XA[3, :], binsA3)

            X21 = digitize(XB[1, :], binsA1)
            X22 = digitize(XB[2, :], binsA2)
            X23 = digitize(XB[3, :], binsA3)

            X11c = to_categorical(X11, 1:2)[2:end, :]
            X21c = to_categorical(X21, 1:2)[2:end, :]
            X12c = to_categorical(X12, 1:3)[2:end, :]
            X22c = to_categorical(X22, 1:3)[2:end, :]
            X13c = to_categorical(X13, 1:4)[2:end, :]
            X23c = to_categorical(X23, 1:4)[2:end, :]

            X1 = vcat(X11c, X12c, X13c)
            X2 = vcat(X21c, X22c, X23c)

            covAemp = cov(X1, dims = 2)
            covBemp = cov(X2, dims = 2)

            aA = params.aA
            aB = params.aB

        else

            X1 = XA
            X2 = XB

            aA = params.aA[1:3]
            aB = params.aB[1:3]

            covAemp = diagm(ones(3))
            covBemp = diagm(ones(3))

        end

        cr2 = 1 / params.r2 - 1


        covA = params.covA
        covB = params.covB

        varerrorA =
            cr2 *
            sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
        varerrorB =
            cr2 *
            sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])

        Base1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), n)
        Base2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), n)

        bYA = quantile(Base1, [0.25, 0.5, 0.75])
        bYB = quantile(Base2, [0.25, 0.5, 0.75])
        bZA = quantile(Base1, [1 / 3, 2 / 3])
        bZB = quantile(Base2, [1 / 3, 2 / 3])


        if scenario == 1
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = copy(binsYA)
            binsZB = copy(binsZA)
        else
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = vcat(-Inf, bYB, Inf)
            binsZB = vcat(-Inf, bZB, Inf)
        end

        return new(
            params,
            binsA1,
            binsA2,
            binsA3,
            covAemp,
            covBemp,
            binsYA,
            binsZA,
            binsYB,
            binsZB,
            discrete,
        )

    end

end


"""
$(SIGNATURES)

Function to generate data where X and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

r2 is the coefficient of determination 

"""
function generate(generator::DataGenerator; eps = 0.0)

    params = generator.params

    dA = MvNormal(params.mA, params.covA)
    XA = rand(dA, params.nA)
    dB = MvNormal(params.mB, params.covB)
    XB = rand(dB, params.nB)

    if generator.discrete

        X11 = digitize(XA[1, :], generator.binsA1)
        X12 = digitize(XA[2, :], generator.binsA2)
        X13 = digitize(XA[3, :], generator.binsA3)

        X21 = digitize(XB[1, :], generator.binsA1)
        X22 = digitize(XB[2, :], generator.binsA2)
        X23 = digitize(XB[3, :], generator.binsA3)

        X11c = to_categorical(X11, 1:2)[2:end, :]
        X21c = to_categorical(X21, 1:2)[2:end, :]
        X12c = to_categorical(X12, 1:3)[2:end, :]
        X22c = to_categorical(X22, 1:3)[2:end, :]
        X13c = to_categorical(X13, 1:4)[2:end, :]
        X23c = to_categorical(X23, 1:4)[2:end, :]

        XX1 = vcat(X11, X21)
        XX2 = vcat(X12, X22)
        XX3 = vcat(X13, X23)

        X1 = vcat(X11c, X12c, X13c)
        X2 = vcat(X21c, X22c, X23c)

    else

        X1 = XA
        X2 = XB

    end

    cr2 = 1.0 / params.r2 - 1


    if generator.discrete

        aA = params.aA
        aB = params.aB
        covA = generator.covAemp
        covB = generator.covBemp

    else

        aA = params.aA[1:3]
        aB = params.aB[1:3]
        covA = params.covA
        covB = params.covB

    end

    σA = cr2 * sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
    σB = cr2 * sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])

    Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(σA)), params.nA)
    Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(σB)), params.nB)

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)

    if generator.discrete
        @info "Categories in XA1 $(sort!(OrderedDict(countmap(X11))))"
        @info "Categories in XA2 $(sort!(OrderedDict(countmap(X12))))"
        @info "Categories in XA3 $(sort!(OrderedDict(countmap(X13))))"
        @info "Categories in XB1 $(sort!(OrderedDict(countmap(X21))))"
        @info "Categories in XB2 $(sort!(OrderedDict(countmap(X22))))"
        @info "Categories in XB3 $(sort!(OrderedDict(countmap(X23))))"
        df = DataFrame(hcat(XX1, XX2, XX3) .- 1, [:X1, :X2, :X3])
    else
        df = DataFrame(hcat(X1, X2)', [:X1, :X2, :X3])
    end

    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))"
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

using Distributions
using DataFrames
import StatsBase: countmap
import OrderedCollections: OrderedDict

digitize(x, bins) = searchsortedlast.(Ref(bins), x)

to_categorical(x) = sort(unique(x)) .== permutedims(x)

to_categorical(x, levels) = levels .== permutedims(x)

export DiscreteDataGenerator

export generate

struct DiscreteDataGenerator

    params::DataParameters
    binsA::Vector{Vector{Float64}}
    covAemp::Matrix{Float64}
    covBemp::Matrix{Float64}
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}

    function DiscreteDataGenerator(params; scenario = 1, n = 10000)

        dA = MvNormal(params.mA, params.covA)
        dB = MvNormal(params.mB, params.covB)

        XA = rand(dA, n)
        XB = rand(dB, n)

        pxcc = [cumsum(pxc)[1:(end - 1)] for pxc in params.pxc]

        qxAc = [quantile.(Normal(params.mA[i], sqrt(params.covA[i, i])), pxcc[i]) for i in eachindex(pxcc)]
        qxBc = [quantile.(Normal(params.mB[i], sqrt(params.covB[i, i])), pxcc[i]) for i in eachindex(pxcc)]

        binsA = [vcat(-Inf, qx, Inf) for qx in qxAc]

        X1 = [digitize(XA[i, :], binsA[i]) for i in eachindex(binsA)]
        X2 = [digitize(XB[i, :], binsA[i]) for i in eachindex(binsA)]

        X1c = [to_categorical(X)[2:end, :] for X in X1]
        X2c = [to_categorical(X)[2:end, :] for X in X2]

        X1 = vcat(X1c...)
        X2 = vcat(X2c...)

        covAemp = cov(X1, dims = 2)
        covBemp = cov(X2, dims = 2)

        cr2 = 1 / params.r2 - 1

        aA = params.aA[1:size(X1,1)]
        aB = params.aB[1:size(X2,1)]

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
            binsA,
            covAemp,
            covBemp,
            binsYA,
            binsZA,
            binsYB,
            binsZB,
        )

    end

end


"""
$(SIGNATURES)

Function to generate data where X covariable are **discrete** and (Y,Z) are categoricals

the function returns a Dataframe with X1, X2, X3, Y, Z and the database id.

r2 is the coefficient of determination 

"""
function generate(generator::DiscreteDataGenerator; eps = 0.0)

    params = generator.params

    dA = MvNormal(params.mA, params.covA)
    XA = rand(dA, params.nA)
    dB = MvNormal(params.mB, params.covB)
    XB = rand(dB, params.nB)

    X1 = [digitize(XA[i, :], generator.binsA[i]) for i in eachindex(generator.binsA)]
    X2 = [digitize(XB[i, :], generator.binsA[i]) for i in eachindex(generator.binsA)]

    X1c = [to_categorical(X)[2:end, :] for X in X1]
    X2c = [to_categorical(X)[2:end, :] for X in X2]

    XX = [vcat(x1, x2) for (x1, x2) in zip(X1, X2)]

    X1 = vcat(X1c...)
    X2 = vcat(X2c...)

    cr2 = 1.0 / params.r2 - 1

    aA = params.aA
    aB = params.aB
    covA = generator.covAemp
    covB = generator.covBemp

    σA = cr2 * sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
    σB = cr2 * sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])

    Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(σA)), params.nA)
    Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(σB)), params.nB)

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)

    columns = [Symbol("X$i") for i in eachindex(XX)]
    df = DataFrame(hcat(XX...) .- 1, columns)

    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))"
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

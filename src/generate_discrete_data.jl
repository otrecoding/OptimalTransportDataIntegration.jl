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
    covA::Matrix{Float64}
    covB::Matrix{Float64}
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}

    function DiscreteDataGenerator(params; scenario = 1, n = 10000)

        q = length(params.mA)

        XA = stack([rand(Categorical(params.pA[i]), params.nA) for i in 1:q], dims = 1)
        XB = stack([rand(Categorical(params.pB[i]), params.nB) for i in 1:q], dims = 1)

        X1 = XA
        X2 = XB

        if q === 1
            covA = fill(cov(vec(XA)), (1, 1))
            covB = fill(cov(vec(XB)), (1, 1))
        else
            covA = cov(XA, dims = 1)
            covB = cov(XB, dims = 1)
        end

        aA = params.aA
        aB = params.aB

        cr2 = 1 / params.r2 - 1

        σA = cr2 * sum([params.aA[i]*aA[j]*cov(XA[i,:], XA[j,:]) for i in 1:q, j in 1:q])
        σB = cr2 * sum([params.aB[i]*aB[j]*cov(XB[i,:], XB[j,:]) for i in 1:q, j in 1:q])
        
        # Base1
        if σA == 0
            Base1 = X1' * params.aA[1:q]  # pas de bruit
        else
            Base1 = X1' * params.aA[1:q] .+ rand(Normal(0.0, sqrt(σA)), params.nA)
        end

# Base2
        if σB == 0
            Base2 = X2' * params.aB[1:q]  # pas de bruit
        else
            Base2 = X2' * params.aB[1:q] .+ rand(Normal(0.0, sqrt(σB)), params.nB)
        end

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
            covA,
            covB,
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

    q = length(params.mA)

    XA = stack([rand(Categorical(params.pA[i]), params.nA) for i in 1:q], dims = 1)
    XB = stack([rand(Categorical(params.pB[i]), params.nB) for i in 1:q], dims = 1)

    X1 = XA
    X2 = XB

    cr2 = 1.0 / params.r2 - 1

    aA = params.aA[1:q]
    aB = params.aB[1:q]

    covA = generator.covA
    covB = generator.covB

    cr2 = 1 / params.r2 - 1
    σA = cr2 * sum([aA[i] * aA[j] * cov(XA[i, :], XA[j, :]) for i in 1:q, j in 1:q])
    σB = cr2 * sum([aB[i] * aB[j] * cov(XB[i, :], XB[j, :]) for i in 1:q, j in 1:q])

    if σA == 0
        Y1 = X1' * aA
    else
        Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(σA)), params.nA)
    end

    if σB == 0
        Y2 = X2' * aB
    else
        Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(σB)), params.nB)
    end

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)

    for j in 1:q
        @info "Categories in XA$j $(sort!(OrderedDict(countmap(XA[j, :]))))"
        @info "Categories in XB$j $(sort!(OrderedDict(countmap(XB[j, :]))))"
    end

    colnames = Symbol.("X" .* string.(1:q))

    df = DataFrame(hcat(X1, X2)', colnames)

    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))" #
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df


end

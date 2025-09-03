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
    covAemp::Matrix{Float64}
    covBemp::Matrix{Float64}
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}
    discrete::Bool

    function DataGenerator_d(params; scenario = 1, nA = 10000, nB = 10000)
          q = length(pA)
          XA=  stack([rand(Categorical(pA[i]), nA) for i in 1:q])
          XB=  stack([rand(Categorical(pB[i]), nB) for i in 1:q])

          X1 = XA
          X2 = XB
          covAemp = cov(XA, dims = 1)
          covBemp = cov(XB, dims = 1)

          aA = params.aA
          aB = params.aB

          cr2 = 1 / params.r2 - 1

          covA = covAemp
          covB = covBemp

          σA = cr2 * sum([aA[i]*aA[j]*cov(XA[:,i], XA[:,j]) for i in 1:q, j in 1:q])
          σB = cr2 * sum([aB[i]*aB[j]*cov(XB[:,i], XB[:,j]) for i in 1:q, j in 1:q])
          #ci dessous ne marche pas !
          Base1 = X1 * aA .+ rand(Normal(0.0, sqrt(σA)), params.nA)
          Base2 = X2 * aB .+ rand(Normal(0.0, sqrt(σB)), params.nB)

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
function generate_d(generator::DataGenerator; eps = 0.0)

    params = generator.params

    q = length(pA)
    XA=  stack([rand(Categorical(pA[i]), nA) for i in 1:q])
    XB=  stack([rand(Categorical(pB[i]), nB) for i in 1:q])
    X1 = XA
    X2 = XB

    cr2 = 1.0 / params.r2 - 1
    aA = params.aA
    aB = params.aB
    covA = generator.covAemp
    covB = generator.covBemp

    cr2 = 1 / r2 - 1
    σA = cr2 * sum([aA[i]*aA[j]*cov(XA[:,i], XA[:,j]) for i in 1:q, j in 1:q])
    σB = cr2 * sum([aB[i]*aB[j]*cov(XB[:,i], XB[:,j]) for i in 1:q, j in 1:q])

    Y1 = X1 * aA .+ rand(Normal(0.0, sqrt(σA)), params.nA)
    Y2 = X2 * aB .+ rand(Normal(0.0, sqrt(σB)), params.nB)

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)

    for j in 1:q
        @info "Categories in XA$j $(sort!(OrderedDict(countmap(XA[:,j]))))"
        @info "Categories in XB$j $(sort!(OrderedDict(countmap(XB[:,j]))))"
    end

    # Construire automatiquement le DataFrame
    colnames = Symbol.("X" .* string.(1:q))  # [:X1, :X2, ..., :Xq]
    df = DataFrame(vcat(X1, X2), colnames)

    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))" #
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

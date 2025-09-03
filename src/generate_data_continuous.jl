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
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}
    discrete::Bool

    function DataGenerator_c(params; scenario = 1,  nA = 10000, nB = 10000,)
    
        XA = rand(MvNormal(params.mA, params.covA),nA)
        XB = rand(MvNormal(params.mB, params.covB),nB)

        X1 = XA
        X2 = XB

        aA = params.aA
        aB = params.aB

        cr2 = 1 / params.r2 - 1

        covA = params.covA
        covB = params.covB
        #il faut que covA et covB soient des matrices, c'est le cas
        varerrorA =
            cr2 *
            sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
        varerrorB =
            cr2 *
            sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])

        Base1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), nA)
        Base2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), nB)

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
function generate_c(generator::DataGenerator; eps = 0.0)

    params = generator.params

   # if length(params.mA) == 1
   #     dA = Normal(params.mA[1], sqrt(params.covA[1,1]))
    #    dB = Normal(params.mB[1], sqrt(params.covB[1,1]))
   # else
    #    dA = MvNormal(params.mA, params.covA)
    #    dB = MvNormal(params.mB, params.covB)
   # end
    XA = rand(MvNormal(params.mA, params.covA),nA)
    XB = rand(MvNormal(params.mB, params.covB),nB)
    X1 = XA
    X2 = XB

    cr2 = 1.0 / params.r2 - 1


    aA = params.aA
    aB = params.aB
    covA = params.covA
    covB = params.covB

    varerrorA =
            cr2 *
            sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
    varerrorB =
            cr2 *
            sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])


    Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
    Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)
    p=length(aA)
    colnames = Symbol.("X" .* string.(1:p))  # [:X1, :X2, ..., :Xp]
    df = DataFrame(vcat(X1', X2'), colnames)
    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))"
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

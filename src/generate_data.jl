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

    function DataGenerator(params; scenario = 1, n = 10000, discrete = true)
    
        if discrete
          #on genere directement q=3 variables discretes
         # pA = [
         #      [0.5, 0.5],                # 2 catégories
         #      [1/3,1/3,1/3],             # 3 catégories
         #      [0.25, 0.25, 0.25, 0.25],  # 4 catégories
         # ]
         # pB = [
         #      [0.7, 0.3],                # 2 catégories
         #      [1/3,1/3,1/3],             # 3 catégories
         #      [0.25, 0.25, 0.25, 0.25],  # 4 catégories
         # ]
          XA= [rand(Categorical(pA[i]), n) for i in 1:q]
          XB= [rand(Categorical(pB[i]), n) for i in 1:q]
        else
        
          dA = MvNormal(params.mA, params.covA) #taille dans vecteur mA
          dB = MvNormal(params.mB, params.covB)

          XA = rand(dA, n)
          XB = rand(dB, n)
        end

        if discrete
            X1 = XA
            X2 = XB
            covAemp = cov(XA, dims = 2)
            covBemp = cov(XB, dims = 2)

            aA = params.aA
            aB = params.aB

        else

            X1 = XA
            X2 = XB

            aA = params.aA
            aB = params.aB

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
        XA= [rand(Categorical(pA[i]), n) for i in 1:q]
        XB= [rand(Categorical(pB[i]), n) for i in 1:q]
        X1 = XA
        X2 = XB

    else
        dA = MvNormal(params.mA, params.covA)
        XA = rand(dA, params.nA)
        dB = MvNormal(params.mB, params.covB)
        XB = rand(dB, params.nB)
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

        aA = params.aA
        aB = params.aB
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
        df = DataFrame(hcat(XX1, XX2, XX3) .- 1, [:X1, :X2, :X3]) #######changer ici pour q variable X
    else
        df = DataFrame(hcat(X1, X2)', [:X1, :X2, :X3]) #######changer ici pour q variable X
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

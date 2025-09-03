#using OptimalTransportDataIntegration

#params = DataParameters(mB = [5, 5, 5], aA = [1.,1,1], aB = [1.,1,1])
#params = DataParameters(mA = [0],mB = [5], covA=[[1]], covB=[[1]], aA = [1.], aB = [1.])


using Distributions
using DataFrames
import StatsBase: countmap
import OrderedCollections: OrderedDict
digitize(x, bins) = searchsortedlast.(Ref(bins), x)

to_categorical(x) = sort(unique(x)) .== permutedims(x)

to_categorical(x, levels) = levels .== permutedims(x)
nA = 1000
nB = 1000
mA = [0.0]
mB= [0.0]
covA =  hcat([1.0])
covB =  hcat([1.0])

aA = [1.0]
aB= [1.0]
r2 = 0.6
XA = rand(MvNormal(mA, covA),nA)
XB = rand(MvNormal(mB, covB),nB)

X1 = XA
X2 = XB

    cr2 = 1.0 / r2 - 1



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
binsYA = vcat(-Inf, bYA, Inf)
binsZA = vcat(-Inf, bZA, Inf)
binsYB = copy(binsYA)
binsZB = copy(binsZA)
     

YA = digitize(Y1, binsYA)
    ZA = digitize(Y1, binsZA)

    YB = digitize(Y2, binsYB )
    ZB = digitize(Y2, binsZB )
    p=length(aA)
    colnames = Symbol.("X" .* string.(1:p))  # [:X1, :X2, ..., :Xp]
    
df = DataFrame(vcat(X1', X2'), colnames)
    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1,nA), fill(2, nB))
    
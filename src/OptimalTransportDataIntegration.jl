module OptimalTransportDataIntegration

using Distributions
using DocStringExtensions
using Parameters
using Printf


export digitize

digitize(x, bins) = searchsortedlast.(Ref(bins), x)

export to_categorical

to_categorical(x) = sort(unique(x)) .== permutedims(x)

to_categorical(x, levels) = levels .== permutedims(x)

# Data generation functions
include("data_parameters.jl")
include("generate_data.jl")
include("generate_xcat_ycat.jl")
include("one_hot_encoder.jl")

# OTRecod functions
include("instance.jl")
include("solution.jl")
include("average_distance_closest.jl")
include("pred_error.jl")
include("otjoint.jl")

# Data integration functions
include("entropic_partial_wasserstein.jl")
include("unbalanced_modality.jl")
include("simple_learning.jl")

# Generic interface
include("otrecod.jl")

export accuracy

accuracy( ypred :: AbstractVector, ytrue :: AbstractVector) = mean( ypred .== ytrue )

function accuracy( data :: DataFrame, yb_pred :: AbstractVector, za_pred :: AbstractVector)
 
    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    Y = vec(data.Y)
    Z = vec(data.Z)

    yb_true = view(Y, indB)
    za_true = view(Z, indA)

    return accuracy(yb_true, yb_pred), accuracy( za_true, za_pred ), accuracy( vcat(yb_pred, za_pred), vcat(yb_true, za_true))

end

end

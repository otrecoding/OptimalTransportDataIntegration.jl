module OptimalTransportDataIntegration

using Distributions
using DocStringExtensions
using Parameters
using Printf

# Data generation functions
include("data_parameters.jl")
include("generate_data_with_p.jl")
include("generate_data_with_r2.jl")
include("one_hot_encoder.jl")

# OTRecod functions
include("instance.jl")
include("solution.jl")
include("average_distance_closest.jl")
include("pred_error.jl")
include("joint_ot_within_base.jl")

# Data integration functions
include("entropic_partial_wasserstein.jl")
include("joint_ot_between_bases.jl")
include("simple_learning.jl")

# Generic interface

struct JointOTResult

    yb_true :: Vector{Int}
    za_true :: Vector{Int}
    yb_pred :: Vector{Int}
    za_pred :: Vector{Int}

end

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


function accuracy(sol :: JointOTResult) 

    accuracy(sol.yb_true, sol.yb_pred), accuracy( sol.za_true, sol.za_pred ), accuracy( vcat(sol.yb_pred, sol.za_pred), vcat(sol.yb_true, sol.za_true))

end

end

module OptimalTransportDataIntegration

using Distributions
using DocStringExtensions
using Parameters
using Printf

include("data_parameters.jl")
include("generate_discrete_data.jl")
include("generate_continuous_data.jl")
include("one_hot_encoder.jl")

include("instance.jl")
include("solution.jl")
include("average_distance_closest.jl")
include("pred_error.jl")
include("joint_ot_within_base_discrete.jl")
include("joint_ot_within_base_continuous.jl")

include("entropic_partial_wasserstein.jl")
include("simple_learning.jl")
include("simple_learning_with_continuous_data.jl")

include("joint_ot_between_bases_discrete.jl")
include("joint_ot_between_bases_category.jl")
include("joint_ot_between_bases_with_predictors.jl")
include("joint_ot_between_bases_without_outcomes.jl")
include("joint_ot_between_bases_jdot.jl")
include("joint_ot_between_bases_da_covariables.jl")
include("joint_ot_between_bases_da_outcomes.jl")
include("joint_ot_between_bases_da_outcomes_with_predictors.jl")

struct JointOTResult

    yb_true::Vector{Int}
    za_true::Vector{Int}
    yb_pred::Vector{Int}
    za_pred::Vector{Int}

end

include("otrecod.jl")

export accuracy

"""
$(SIGNATURES)
"""
accuracy(ypred::AbstractVector, ytrue::AbstractVector) = mean(ypred .== ytrue)

"""
$(SIGNATURES)
"""
function accuracy(data::DataFrame, yb_pred::AbstractVector, za_pred::AbstractVector)

    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    Y = vec(data.Y)
    Z = vec(data.Z)

    yb_true = view(Y, indB)
    za_true = view(Z, indA)

    return accuracy(yb_true, yb_pred),
        accuracy(za_true, za_pred),
        accuracy(vcat(yb_pred, za_pred), vcat(yb_true, za_true))

end


"""
$(SIGNATURES)
"""
function accuracy(sol::JointOTResult)

    return accuracy(sol.yb_true, sol.yb_pred),
        accuracy(sol.za_true, sol.za_pred),
        accuracy(vcat(sol.yb_pred, sol.za_pred), vcat(sol.yb_true, sol.za_true))

end

export confusion_matrix

"""
$(SIGNATURES)

made by claude.ai
"""
function confusion_matrix(y_true, y_pred; classes = 1:4)

    n_classes = length(classes)

    class_to_idx = Dict(class => i for (i, class) in enumerate(classes))

    cm = zeros(Int, n_classes, n_classes)

    for (true_label, pred_label) in zip(y_true, y_pred)
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    end

    return cm
end

end

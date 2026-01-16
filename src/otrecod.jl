"""
    otrecod.jl

Core dispatcher module for statistical data matching methods based on optimal transport theory.

This module defines the abstract method interface and concrete implementations for various
approaches to statistical matching between data sources. All methods follow a uniform interface
via the `otrecod()` dispatcher function.
"""

using Parameters

export otrecod

"""
    AbstractMethod

Abstract base type for all statistical matching methods.

All concrete methods (e.g., `JointOTWithinBase`, `SimpleLearning`) must inherit from this type
and implement a corresponding `otrecod(data::DataFrame, method::YourMethod)` function.
"""
abstract type AbstractMethod end

export JointOTWithinBase

"""
    JointOTWithinBase <: AbstractMethod

Statistical matching using within-base optimal transport balancing.

This method balances the joint distribution of covariates and outcomes within each data source
separately before making predictions. It works with both discrete and continuous covariates.

# Fields
- `lambda::Float64 = 0.1`: Regularization parameter for transport plan
- `alpha::Float64 = 0.1`: Weight parameter for balancing
- `percent_closest::Float64 = 0.2`: Fraction of closest neighbors to consider
- `distance::Distances.Metric = Euclidean()`: Distance metric for covariate space
"""
@with_kw struct JointOTWithinBase <: AbstractMethod

    lambda::Float64 = 0.1
    alpha::Float64 = 0.1
    percent_closest::Float64 = 0.2
    distance::Distances.Metric = Euclidean()

end

"""
    otrecod(data::DataFrame, method::JointOTWithinBase)

Apply within-base optimal transport balancing to match data sources.

Automatically detects if covariates are discrete or continuous and applies the appropriate
algorithm. Returns predictions for missing outcomes (Y in base A, Z in base B).

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 or 2), `X*` (covariates), `Y`, `Z` (outcomes)
- `method::JointOTWithinBase`: Method configuration

# Returns
- `JointOTResult`: Tuple containing (yb_true, za_true, yb_pred, za_pred)
"""
function otrecod(data::DataFrame, method::JointOTWithinBase)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    if discrete

        yb_pred, za_pred = joint_ot_within_base_discrete(
            data;
            lambda = method.lambda,
            alpha = method.alpha,
            percent_closest = method.percent_closest,
            distance = method.distance,
        )

    else

        yb_pred, za_pred = joint_ot_within_base_continuous(
            data;
            lambda = method.lambda,
            alpha = method.alpha,
            percent_closest = method.percent_closest,
            distance = method.distance,
        )

    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export SimpleLearning

"""
    SimpleLearning <: AbstractMethod

Neural network-based statistical matching using Flux.jl.

Uses a feedforward neural network with one hidden layer to learn the relationship between
covariates and outcomes. Simple baseline for comparison with optimal transport methods.

# Fields
- `hidden_layer_size::Int = 10`: Number of neurons in hidden layer
- `learning_rate::Float64 = 0.01`: Learning rate for optimization
- `batchsize::Int = 64`: Mini-batch size for training
- `epochs::Int = 1000`: Number of training epochs
"""
@with_kw struct SimpleLearning <: AbstractMethod

    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::SimpleLearning)

Apply neural network-based learning for statistical matching.

Automatically routes to discrete or continuous implementation based on covariate types.
Trains separate networks for each outcome variable.

# Arguments
- `data::DataFrame`: Input data with columns `database`, `X*`, `Y`, `Z`
- `method::SimpleLearning`: Learning configuration

# Returns
- `JointOTResult`: Predictions and ground truth values
"""
function otrecod(data::DataFrame, method::SimpleLearning)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    if discrete
        yb_pred, za_pred = simple_learning(
            data;
            hidden_layer_size = method.hidden_layer_size,
            learning_rate = method.learning_rate,
            batchsize = method.batchsize,
            epochs = method.epochs,
        )
    else
        yb_pred, za_pred = learning_with_continuous_data(
            data;
            hidden_layer_size = method.hidden_layer_size,
            learning_rate = method.learning_rate,
            batchsize = method.batchsize,
            epochs = method.epochs,
        )
    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTBetweenBases

"""
    JointOTBetweenBases <: AbstractMethod

Joint optimal transport between data sources with discrete outcomes.

Solves optimal transport problems that match the joint distribution of covariates and outcomes
from base B to base A. Requires discrete (categorical) covariates.

# Fields
- `reg::Float64 = 0.001`: Entropy regularization parameter (0 = exact OT)
- `reg_m1::Float64 = 0.01`: Marginal constraint relaxation for first moment
- `reg_m2::Float64 = 0.01`: Marginal constraint relaxation for second moment
- `Ylevels::Vector{Int} = 1:4`: Discrete levels of outcome Y
- `Zlevels::Vector{Int} = 1:3`: Discrete levels of outcome Z
- `iterations::Int = 10`: Number of Sinkhorn iterations
"""
@with_kw struct JointOTBetweenBases <: AbstractMethod
    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
end

"""
    otrecod(data::DataFrame, method::JointOTBetweenBases)

Apply joint optimal transport between discrete data sources.

Requires covariates to be categorical. Matches distributions between bases and
predicts missing outcomes using the transport map.

# Arguments
- `data::DataFrame`: Input data with discrete covariates and outcomes
- `method::JointOTBetweenBases`: OT configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are not categorical
"""
function otrecod(data::DataFrame, method::JointOTBetweenBases)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert discrete "JointOTBetweenBases: covariables must be categorical"

    yb_pred, za_pred = joint_ot_between_bases_discrete(
        data,
        method.reg,
        method.reg_m1,
        method.reg_m2;
        Ylevels = method.Ylevels,
        Zlevels = method.Zlevels,
        iterations = method.iterations
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTBetweenBasesDiscreteOrdered

"""
$(TYPEDEF)

Joint optimal transport between data sources with categorical outcomes.

Extension of `JointOTBetweenBases` using one-hot encoding for categorical outcomes.
Enables better handling of discrete outcome distributions via explicit category representation.

# Fields
- `reg::Float64 = 0.001`: Entropy regularization parameter
- `reg_m1::Float64 = 0.01`: First moment marginal relaxation
- `reg_m2::Float64 = 0.01`: Second moment marginal relaxation
- `Ylevels::Vector{Int} = 1:4`: Categories of outcome Y
- `Zlevels::Vector{Int} = 1:3`: Categories of outcome Z
- `iterations::Int = 10`: Sinkhorn iterations
"""
@with_kw struct JointOTBetweenBasesDiscreteOrdered <: AbstractMethod
    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
end

"""
    otrecod(data::DataFrame, method::JointOTBetweenBasesDiscreteOrdered)

Apply joint OT with categorical outcome encoding.

Similar to `JointOTBetweenBases` but uses one-hot encoding for categorical outcomes,
which can improve numerical stability and interpretability.

# Arguments
- `data::DataFrame`: Input data with discrete covariates
- `method::JointOTBetweenBasesDiscreteOrdered`: Configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are not categorical
"""
function otrecod(data::DataFrame, method::JointOTBetweenBasesDiscreteOrdered)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert discrete "JointOTBetweenBasesDiscreteOrdered: covariables must be categorical"

    yb_pred, za_pred = joint_ot_between_bases_discrete_ordered(
        data,
        method.reg,
        method.reg_m1,
        method.reg_m2;
        Ylevels = method.Ylevels,
        Zlevels = method.Zlevels,
        iterations = method.iterations
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTBetweenBasesWithPredictors

"""
    JointOTBetweenBasesWithPredictors <: AbstractMethod

Hybrid method combining optimal transport with neural network predictors.

Combines OT-based covariate matching with neural networks to predict outcomes.
Enables handling of both discrete and continuous covariates.

# Fields
- `reg::Float64 = 0.001`: OT entropy regularization
- `reg_m1::Float64 = 0.01`: Marginal relaxation (first moment)
- `reg_m2::Float64 = 0.01`: Marginal relaxation (second moment)
- `Ylevels::Vector{Int} = 1:4`: Y outcome categories
- `Zlevels::Vector{Int} = 1:3`: Z outcome categories
- `iterations::Int = 10`: Transport iterations
- `hidden_layer_size::Int = 10`: Neural network hidden layer size
- `learning_rate::Float64 = 0.01`: Learning rate for network training
- `batchsize::Int = 64`: Mini-batch size
- `epochs::Int = 1000`: Training epochs
"""
@with_kw struct JointOTBetweenBasesWithPredictors <: AbstractMethod

    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::JointOTBetweenBasesWithPredictors)

Apply hybrid OT + neural network approach.

Combines transport plan computation with network-based outcome prediction.
Works with both discrete and continuous covariates.

# Arguments
- `data::DataFrame`: Input data with outcomes
- `method::JointOTBetweenBasesWithPredictors`: Hybrid configuration

# Returns
- `JointOTResult`: Predictions and ground truth
"""
function otrecod(data::DataFrame, method::JointOTBetweenBasesWithPredictors)


    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    yb_pred, za_pred = joint_ot_between_bases_with_predictors(
        data;
        iterations = method.iterations,
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
        Ylevels = method.Ylevels,
        Zlevels = method.Zlevels,
        reg = method.reg,
        reg_m1 = method.reg_m1,
        reg_m2 = method.reg_m2
    )


    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end


export JointOTBetweenBasesJDOT

"""
    JointOTBetweenBasesJDOT <: AbstractMethod

Joint Distribution Optimal Transport between continuous data sources.

Implements JDOT (Joint Distribution Optimal Transport) which simultaneously learns
a classifier and aligns distributions between sources.

# Fields
- `reg::Float64 = 0.001`: OT entropy regularization
- `reg_m1::Float64 = 0.01`: First moment marginal relaxation
- `reg_m2::Float64 = 0.01`: Second moment marginal relaxation
- `Ylevels::Vector{Int} = 1:4`: Y categories
- `Zlevels::Vector{Int} = 1:3`: Z categories
- `iterations::Int = 10`: OT iterations
- `distance::Distances.Metric = Euclidean()`: Distance metric
- `hidden_layer_size::Int = 10`: Network hidden units
- `learning_rate::Float64 = 0.01`: Training rate
- `batchsize::Int = 64`: Batch size
- `epochs::Int = 1000`: Training epochs
"""
@with_kw struct JointOTBetweenBasesJDOT <: AbstractMethod

    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Euclidean()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::JointOTBetweenBasesJDOT)

Apply JDOT for continuous data sources.

Simultaneously learns outcome predictions and aligns covariate distributions.
Requires continuous covariates.

# Arguments
- `data::DataFrame`: Input data with continuous covariates
- `method::JointOTBetweenBasesJDOT`: JDOT configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are discrete
"""
function otrecod(data::DataFrame, method::JointOTBetweenBasesJDOT)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert !discrete

    yb_pred, za_pred = joint_ot_between_bases_jdot(
        data;
        iterations = method.iterations,
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
        reg = method.reg,
        reg_m1 = method.reg_m1,
        reg_m2 = method.reg_m2
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end


export JointOTDABetweenBasesCovariables

"""
    JointOTDABetweenBasesCovariables <: AbstractMethod

Domain adaptation focusing on covariate alignment.

Approach emphasizing alignment of covariate distributions between sources
using domain adaptation techniques combined with optimal transport.

# Fields
- `reg::Float64 = 0.001`: Entropy regularization
- `reg_m1::Float64 = 0.01`: First moment marginal relaxation
- `reg_m2::Float64 = 0.01`: Second moment marginal relaxation
- `Ylevels::Vector{Int} = 1:4`: Y categories
- `Zlevels::Vector{Int} = 1:3`: Z categories
- `distance::Distances.Metric = Euclidean()`: Metric for covariates
- `hidden_layer_size::Int = 10`: Adapter network hidden size
- `learning_rate::Float64 = 0.01`: Training rate
- `batchsize::Int = 64`: Batch size
- `epochs::Int = 1000`: Training epochs
"""
@with_kw struct JointOTDABetweenBasesCovariables <: AbstractMethod

    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    distance::Distances.Metric = Euclidean()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::JointOTDABetweenBasesCovariables)

Apply domain adaptation with covariate focus.

Aligns covariate distributions while learning outcome predictions.
Requires continuous covariates.

# Arguments
- `data::DataFrame`: Input data with continuous covariates
- `method::JointOTDABetweenBasesCovariables`: DA configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are discrete
"""
function otrecod(data::DataFrame, method::JointOTDABetweenBasesCovariables)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert !discrete

    yb_pred, za_pred = joint_ot_between_bases_da_covariables(
        data;
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
        reg = method.reg,
        reg_m1 = method.reg_m1,
        reg_m2 = method.reg_m2
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end


export JointOTDABetweenBasesOutcomes

"""
    JointOTDABetweenBasesOutcomes <: AbstractMethod

Domain adaptation focusing on outcome variable alignment.

Approach emphasizing alignment of outcome distributions between sources.
Combines OT-based distribution matching with domain adaptation principles.

# Fields
- `reg::Float64 = 0.001`: Entropy regularization
- `reg_m1::Float64 = 0.01`: First moment relaxation
- `reg_m2::Float64 = 0.01`: Second moment relaxation
- `Ylevels::Vector{Int} = 1:4`: Y categories
- `Zlevels::Vector{Int} = 1:3`: Z categories
- `iterations::Int = 10`: OT iterations
- `distance::Distances.Metric = Euclidean()`: Covariate metric
- `hidden_layer_size::Int = 10`: Network hidden size
- `learning_rate::Float64 = 0.01`: Training rate
- `batchsize::Int = 64`: Batch size
- `epochs::Int = 1000`: Training epochs
"""
@with_kw struct JointOTDABetweenBasesOutcomes <: AbstractMethod

    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Euclidean()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::JointOTDABetweenBasesOutcomes)

Apply domain adaptation with outcome focus.

Aligns outcome distributions while performing statistical matching.
Requires continuous covariates.

# Arguments
- `data::DataFrame`: Input data with continuous covariates
- `method::JointOTDABetweenBasesOutcomes`: DA configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are discrete
"""
function otrecod(data::DataFrame, method::JointOTDABetweenBasesOutcomes)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert !discrete

    yb_pred, za_pred = joint_ot_between_bases_da_outcomes(
        data;
        iterations = method.iterations,
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
        reg = method.reg,
        reg_m1 = method.reg_m1,
        reg_m2 = method.reg_m2
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTDABetweenBasesOutcomesWithPredictors

"""
    JointOTDABetweenBasesOutcomesWithPredictors <: AbstractMethod

Domain adaptation with outcome alignment and integrated neural predictors.

Combines outcome distribution alignment, domain adaptation, and neural network-based
outcome prediction. Fully hybrid approach suitable for continuous data.

# Fields
- `reg::Float64 = 0.001`: Entropy regularization
- `reg_m1::Float64 = 0.01`: First moment relaxation
- `reg_m2::Float64 = 0.01`: Second moment relaxation
- `Ylevels::Vector{Int} = 1:4`: Y categories
- `Zlevels::Vector{Int} = 1:3`: Z categories
- `iterations::Int = 10`: OT iterations
- `hidden_layer_size::Int = 10`: Predictor network hidden size
- `learning_rate::Float64 = 0.01`: Training rate
- `batchsize::Int = 64`: Batch size
- `epochs::Int = 1000`: Training epochs
"""
@with_kw struct JointOTDABetweenBasesOutcomesWithPredictors <: AbstractMethod

    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::JointOTDABetweenBasesOutcomesWithPredictors)

Apply full domain adaptation with outcome alignment and predictors.

Integrates outcome distribution alignment with neural network prediction.
Requires continuous covariates.

# Arguments
- `data::DataFrame`: Input data with continuous covariates
- `method::JointOTDABetweenBasesOutcomesWithPredictors`: Configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are discrete
"""
function otrecod(data::DataFrame, method::JointOTDABetweenBasesOutcomesWithPredictors)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert !discrete

    yb_pred, za_pred = joint_ot_between_bases_da_outcomes_with_predictors(
        data;
        iterations = method.iterations,
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
        reg = method.reg,
        reg_m1 = method.reg_m1,
        reg_m2 = method.reg_m2
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTBetweenBasesWithoutOutcomes

"""
    JointOTBetweenBasesWithoutOutcomes <: AbstractMethod

Statistical matching using OT without observed outcome variables.

Variant that performs matching based solely on covariate distributions when outcome
variables are not fully observed. Uses domain adaptation to learn implicit outcome predictors.

# Fields
- `reg::Float64 = 0.001`: Entropy regularization
- `reg_m1::Float64 = 0.01`: First moment relaxation
- `reg_m2::Float64 = 0.01`: Second moment relaxation
- `Ylevels::Vector{Int} = 1:4`: Expected Y categories
- `Zlevels::Vector{Int} = 1:3`: Expected Z categories
- `iterations::Int = 10`: OT iterations
- `hidden_layer_size::Int = 10`: Implicit predictor network hidden size
- `learning_rate::Float64 = 0.01`: Training rate
- `batchsize::Int = 64`: Batch size
- `epochs::Int = 1000`: Training epochs
"""
@with_kw struct JointOTBetweenBasesWithoutOutcomes <: AbstractMethod

    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

"""
    otrecod(data::DataFrame, method::JointOTBetweenBasesWithoutOutcomes)

Apply OT-based matching without explicit outcome observations.

Matches distributions and predicts outcomes using implicit predictors learned
from covariate structure. Requires continuous covariates.

# Arguments
- `data::DataFrame`: Input data (outcomes may be partially missing)
- `method::JointOTBetweenBasesWithoutOutcomes`: Configuration

# Returns
- `JointOTResult`: Predictions and ground truth

# Throws
- `AssertionError`: If covariates are discrete
"""
function otrecod(data::DataFrame, method::JointOTBetweenBasesWithoutOutcomes)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert !discrete

    yb_pred, za_pred = joint_ot_between_bases_without_outcomes(
        data;
        iterations = method.iterations,
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
        reg = method.reg,
        reg_m1 = method.reg_m1,
        reg_m2 = method.reg_m2
    )

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

using Parameters

export otrecod

abstract type AbstractMethod end

export JointOTWithinBase

@with_kw struct JointOTWithinBase <: AbstractMethod

    lambda::Float64 = 0.1
    alpha::Float64 = 0.1
    percent_closest::Float64 = 0.2
    distance::Distances.Metric = Euclidean()

end

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

@with_kw struct SimpleLearning <: AbstractMethod

    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

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

@with_kw struct JointOTBetweenBases <: AbstractMethod
    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
end

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

export JointOTBetweenBasesCategory

@with_kw struct JointOTBetweenBasesCategory <: AbstractMethod
    reg::Float64 = 0.001
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::Vector{Int} = 1:4
    Zlevels::Vector{Int} = 1:3
    iterations::Int = 10
end

function otrecod(data::DataFrame, method::JointOTBetweenBasesCategory)

    xcols = names(data, r"^X")
    discrete = all(isinteger.(Matrix(data[!, xcols])))

    @assert discrete "JointOTBetweenBasesCategory: covariables must be categorical"

    yb_pred, za_pred = joint_ot_between_bases_category(
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

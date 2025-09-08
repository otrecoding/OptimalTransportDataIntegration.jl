using Parameters

export otrecod

abstract type AbstractMethod end

export JointOTWithinBase

@with_kw struct JointOTWithinBase <: AbstractMethod

    lambda::Float64 = 0.1
    alpha::Float64 = 0.1
    percent_closest::Float64 = 0.2
    distance::Distances.Metric = Hamming()

end

function otrecod(data::DataFrame, method::JointOTWithinBase)

    discrete = all(isinteger.(data.X1))

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

    discrete = all(isinteger.(data.X1))

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

    reg::Float64 = 0.01
    reg_m1::Float64 = 0.05
    reg_m2::Float64 = 0.05
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Hamming()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::JointOTBetweenBases)

    discrete = all(isinteger.(data.X1))

    if discrete

        yb_pred, za_pred = joint_ot_between_bases(
            data,
            method.reg,
            method.reg_m1,
            method.reg_m2;
            Ylevels = method.Ylevels,
            Zlevels = method.Zlevels,
            iterations = method.iterations,
            distance = method.distance,
        )
    else
        yb_pred, za_pred = joint_between_with_predictors(
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

    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end


export JointOTBetweenBasesRefJDOT

@with_kw struct JointOTBetweenBasesRefJDOT <: AbstractMethod

    reg::Float64 = 0.01
    reg_m1::Float64 = 0.05
    reg_m2::Float64 = 0.05
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Hamming()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::JointOTBetweenBasesRefJDOT)

    discrete = all(isinteger.(data.X1))

    if discrete

        yb_pred, za_pred = joint_ot_between_bases(
            data,
            method.reg,
            method.reg_m1,
            method.reg_m2;
            Ylevels = method.Ylevels,
            Zlevels = method.Zlevels,
            iterations = method.iterations,
            distance = method.distance,
        )
    else
        yb_pred, za_pred = joint_between_ref_jdot(
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

    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end


export JointOTBetweenBasesRefOTDAx

@with_kw struct JointOTBetweenBasesRefOTDAx <: AbstractMethod

    reg::Float64 = 0.01
    reg_m1::Float64 = 0.05
    reg_m2::Float64 = 0.05
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Hamming()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::JointOTBetweenBasesRefOTDAx)

    discrete = all(isinteger.(data.X1))

    if discrete

        yb_pred, za_pred = joint_ot_between_bases(
            data,
            method.reg,
            method.reg_m1,
            method.reg_m2;
            Ylevels = method.Ylevels,
            Zlevels = method.Zlevels,
            iterations = method.iterations,
            distance = method.distance,
        )
    else
        yb_pred, za_pred = joint_between_ref_otda_x(
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

    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end


export JointOTBetweenBasesRefOTDAyz

@with_kw struct JointOTBetweenBasesRefOTDAyz <: AbstractMethod

    reg::Float64 = 0.01
    reg_m1::Float64 = 0.05
    reg_m2::Float64 = 0.05
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Hamming()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::JointOTBetweenBasesRefOTDAyz)

    discrete = all(isinteger.(data.X1))

    if discrete

        yb_pred, za_pred = joint_ot_between_bases(
            data,
            method.reg,
            method.reg_m1,
            method.reg_m2;
            Ylevels = method.Ylevels,
            Zlevels = method.Zlevels,
            iterations = method.iterations,
            distance = method.distance,
        )
    else
        yb_pred, za_pred = joint_between_ref_otda_yz(
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

    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTBetweenBasesRefOTDAyzPred

@with_kw struct JointOTBetweenBasesRefOTDAyzPred <: AbstractMethod

    reg::Float64 = 0.01
    reg_m1::Float64 = 0.05
    reg_m2::Float64 = 0.05
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3
    iterations::Int = 10
    distance::Distances.Metric = Hamming()
    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::JointOTBetweenBasesRefOTDAyzPred)

    discrete = all(isinteger.(data.X1))

    if discrete

        yb_pred, za_pred = joint_ot_between_bases(
            data,
            method.reg,
            method.reg_m1,
            method.reg_m2;
            Ylevels = method.Ylevels,
            Zlevels = method.Zlevels,
            iterations = method.iterations,
            distance = method.distance,
        )
    else
        yb_pred, za_pred = joint_between_ref_otda_yz_pred(
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

    end

    yb_true = data.Y[data.database .== 2]
    za_true = data.Z[data.database .== 1]

    return JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

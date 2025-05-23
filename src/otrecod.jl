using Parameters

export otrecod

abstract type AbstractMethod end

export JointOTWithinBase

@with_kw struct JointOTWithinBase <: AbstractMethod

    lambda::Float64 = 0.392
    alpha::Float64 = 0.714
    percent_closest::Float64 = 0.2

end

function otrecod(data::DataFrame, method::JointOTWithinBase)

    yb_pred, za_pred = joint_ot_within_base(
        data;
        lambda = method.lambda,
        alpha = method.alpha,
        percent_closest = method.percent_closest,
    )

    yb_true = data.Y[data.database.==2]
    za_true = data.Z[data.database.==1]

    JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export SimpleLearning

@with_kw struct SimpleLearning <: AbstractMethod

    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::SimpleLearning)

    yb_pred, za_pred = simple_learning(
        data;
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
    )

    yb_true = data.Y[data.database.==2]
    za_true = data.Z[data.database.==1]

    JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

export JointOTBetweenBases

@with_kw struct JointOTBetweenBases <: AbstractMethod

    reg::Float64 = 0.01
    reg_m1::Float64 = 0.01
    reg_m2::Float64 = 0.01
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3
    iterations::Int = 10

end

function otrecod(data::DataFrame, method::JointOTBetweenBases)

    yb_pred, za_pred = joint_ot_between_bases(
        data,
        method.reg,
        method.reg_m1,
        method.reg_m2;
        Ylevels = method.Ylevels,
        Zlevels = method.Zlevels,
        iterations = method.iterations,
    )

    yb_true = data.Y[data.database.==2]
    za_true = data.Z[data.database.==1]

    JointOTResult(yb_true, za_true, yb_pred, za_pred)

end

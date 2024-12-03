using Parameters

export otrecod

abstract type AbstractMethod end

export OTjoint

@with_kw struct OTjoint <: AbstractMethod

    lambda_reg::Float64 = 0.392
    maxrelax::Float64 = 0.714
    percent_closest::Float64 = 0.2

end

function otrecod(data::DataFrame, method::OTjoint)

    otjoint(
        data;
        lambda_reg = method.lambda_reg,
        maxrelax = method.maxrelax,
        percent_closest = method.percent_closest,
    )

end

export SimpleLearning

@with_kw struct SimpleLearning <: AbstractMethod

    hidden_layer_size::Int = 10
    learning_rate::Float64 = 0.01
    batchsize::Int = 64
    epochs::Int = 1000

end

function otrecod(data::DataFrame, method::SimpleLearning)

    simple_learning(
        data;
        hidden_layer_size = method.hidden_layer_size,
        learning_rate = method.learning_rate,
        batchsize = method.batchsize,
        epochs = method.epochs,
    )

end

export UnbalancedModality

@with_kw struct UnbalancedModality <: AbstractMethod

    reg_m::Float64 = 0.1
    reg::Float64 = 0.0
    Ylevels::AbstractVector = 1:4
    Zlevels::AbstractVector = 1:3

end

function otrecod(data::DataFrame, method::UnbalancedModality)

    unbalanced_modality(data, method.reg, method.reg_m; Ylevels = method.Ylevels, Zlevels = method.Zlevels)

end

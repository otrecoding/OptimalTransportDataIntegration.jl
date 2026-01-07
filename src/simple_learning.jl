import Flux
import Flux: Chain, Dense

"""
    onehot(x::AbstractMatrix)

Convert categorical covariates to one-hot encoded matrix format.

For each column in input matrix, creates binary vectors for each level (1:4),
where `col .== level` produces a binary vector. Stacks all one-hot vectors
into rows of output matrix for use in neural network input.

# Arguments
- `x::AbstractMatrix`: Categorical data matrix (n_samples, n_covariates)

# Returns
- `Matrix{Float32}`: One-hot encoded matrix (n_samples * n_levels, n_covariates)
  where n_levels = 4 (hardcoded for discrete covariates)
"""
function onehot(x::AbstractMatrix)
    res = Vector{Float32}[]
    for col in eachcol(x)
        levels = 1:4 #filter(x -> x != 0, sort(unique(col)))
        for level in levels
            push!(res, col .== level)
        end
    end
    return stack(res, dims = 1)
end

"""
    simple_learning(data; hidden_layer_size=10, learning_rate=0.01, batchsize=512, epochs=1000, Ylevels=1:4, Zlevels=1:3)

Statistical matching baseline via supervised neural networks.

Trains two independent 2-layer neural networks: one predicting Y (outcome for base B) 
from X in base A, and one predicting Z (outcome for base A) from X in base B. 
This reference method ignores joint distribution structure, serving as a comparison 
point to validate optimal transport improvements.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
- `hidden_layer_size::Int`: Number of neurons in hidden layer; default: 10
- `learning_rate::Float64`: Adam optimizer learning rate; default: 0.01
- `batchsize::Int`: Batch size for stochastic gradient descent; default: 512
- `epochs::Int`: Training epochs for each network; default: 1000
- `Ylevels::AbstractRange`: Categorical levels for outcome Y; default: 1:4
- `Zlevels::AbstractRange`: Categorical levels for outcome Z; default: 1:3

# Returns
- `Tuple{Vector{Int}, Vector{Int}}`: Predicted outcomes (YB, ZA)
  - `YB`: Predictions for Y in base B (argmax of network outputs)
  - `ZA`: Predictions for Z in base A (argmax of network outputs)

# Model Architecture
Two identical networks:
```
input (one-hot encoded X) → Dense(hidden_layer_size) → Dense(n_levels) → softmax → argmax
```

# Algorithm
1. Split data by database indicator (base A, base B)
2. One-hot encode covariates X for each base
3. One-hot encode outcomes Y and Z with specified levels
4. Train modelA on (XA, YA) and modelB on (XB, ZB) with logit cross-entropy loss
5. Predict YB = argmax(modelA(XB)) and ZA = argmax(modelB(XA))

# See Also
- `JointOTBetweenBases`: OT-based method that captures joint distributions
- `JointOTWithinBase`: Within-source distribution balancing

# Notes
- Independent networks ignore covariate-outcome correlation structure across bases
- Serves as a performance baseline for validating OT methods
- Uses Flux.jl for neural network training with Adam optimizer
"""
function simple_learning(
        data;
        hidden_layer_size = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 1000,
        Ylevels = 1:4,
        Zlevels = 1:3
    )

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    colnames = names(data, r"^X")
    XA = onehot(Matrix(dba[!, colnames]))
    XB = onehot(Matrix(dbb[!, colnames]))

    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

    function train!(model, x, y)

        loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
        optim = Flux.setup(Flux.Adam(learning_rate), model)

        for epoch in 1:epochs
            for (x, y) in loader
                grads = Flux.gradient(model) do m
                    y_hat = m(x)
                    Flux.logitcrossentropy(y_hat, y)
                end
                Flux.update!(optim, model, grads[1])
            end
        end

        return
    end

    train!(modelA, XA, YA)
    train!(modelB, XB, ZB)

    YB = Flux.onecold(modelA(XB))
    ZA = Flux.onecold(modelB(XA))

    return YB, ZA

end

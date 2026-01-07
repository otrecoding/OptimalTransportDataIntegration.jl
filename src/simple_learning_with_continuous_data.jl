import Flux: Chain, Dense

"""
    learning_with_continuous_data(data; hidden_layer_size=10, learning_rate=0.01, batchsize=128, epochs=1000)

Statistical matching baseline via supervised neural networks on continuous covariates.

Trains two independent 2-layer neural networks: one predicting Y (outcome for base B) 
from continuous X in base A, and one predicting Z (outcome for base A) from continuous X 
in base B. This reference method ignores the joint distribution structure, serving as a 
comparison point to validate optimal transport improvements for continuous data.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` continuous covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
- `hidden_layer_size::Int`: Number of neurons in hidden layer; default: 10
- `learning_rate::Float64`: Adam optimizer learning rate; default: 0.01
- `batchsize::Int`: Batch size for stochastic gradient descent; default: 128
- `epochs::Int`: Training epochs for each network; default: 1000

# Returns
- `Tuple{Vector{Int}, Vector{Int}}`: Predicted outcomes (YB, ZA)
  - `YB`: Predictions for Y in base B (argmax of network outputs, categories 1:4)
  - `ZA`: Predictions for Z in base A (argmax of network outputs, categories 1:3)

# Model Architecture
Two identical networks with continuous input:
```
input (continuous X) → Dense(input_dim → hidden_layer_size) → Dense(hidden_layer_size → n_levels) → softmax → argmax
```

# Algorithm
1. Split data by database indicator (base A, base B)
2. Extract continuous covariates X (no one-hot encoding needed)
3. One-hot encode outcomes Y and Z with specified levels (Y: 1:4, Z: 1:3)
4. Train modelA on (XA, YA) and modelB on (XB, ZB) with logit cross-entropy loss
5. Predict YB = argmax(modelA(XB)) and ZA = argmax(modelB(XA))

# Differences from `simple_learning` (discrete version)
- Input: continuous covariates (no one-hot encoding)
- Network input dimension: number of continuous features (not one-hot expanded)
- Uses transposed covariate matrix for Flux compatibility
- Otherwise identical supervised learning approach

# See Also
- `simple_learning`: Discrete covariates version with one-hot encoded X
- `JointOTBetweenBases`: OT-based method that captures joint distributions
- `joint_ot_between_bases_with_predictors`: Hybrid OT + neural network approach

# Notes
- Independent networks ignore covariate-outcome correlation structure across bases
- Serves as a performance baseline for validating OT methods on continuous data
- Uses Flux.jl for neural network training with Adam optimizer
- More computationally efficient than discrete version due to smaller input dimension
"""
function learning_with_continuous_data(
        data;
        hidden_layer_size = 10,
        learning_rate = 0.01,
        batchsize = 128,
        epochs = 1000,
    )

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    colnames = names(data, r"^X")

    XA = transpose(Matrix{Float32}(dba[!, colnames]))
    XB = transpose(Matrix{Float32}(dbb[!, colnames]))

    YA = Flux.onehotbatch(dba.Y, 1:4)
    ZB = Flux.onehotbatch(dbb.Z, 1:3)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    nA = size(XA, 2)
    nB = size(XB, 2)

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

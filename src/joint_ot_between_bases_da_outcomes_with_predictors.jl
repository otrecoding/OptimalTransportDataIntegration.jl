"""
    joint_ot_between_bases_da_outcomes_with_predictors(data; iterations=10, learning_rate=0.01, 
                                                        batchsize=512, epochs=500, hidden_layer_size=10, 
                                                        reg=0.0, reg_m1=0.0, reg_m2=0.0, 
                                                        Ylevels=1:4, Zlevels=1:3)

Advanced discriminant analysis: OT on covariate space with outcome-based predictor refinement.

Solves a single optimal transport problem on covariate-only distance, then trains neural networks
to predict outcomes from transported samples. This "DA with predictors" variant uses the OT coupling
to transport outcome distributions and trains networks on transported outcome targets (not iterative).
Combines covariate-based matching with outcome prediction flexibility.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
- `iterations::Int`: Unused parameter (kept for API compatibility); default: 10
- `learning_rate::Float64`: Adam optimizer learning rate for networks; default: 0.01
- `batchsize::Int`: Batch size for stochastic gradient descent; default: 512
- `epochs::Int`: Training epochs per network; default: 500
- `hidden_layer_size::Int`: Number of neurons in hidden layer; default: 10
- `reg::Float64`: Entropy regularization for OT (0 = exact, larger = relaxed); default: 0.0
- `reg_m1::Float64`: Marginal relaxation for base A; default: 0.0
- `reg_m2::Float64`: Marginal relaxation for base B; default: 0.0
- `Ylevels::AbstractRange`: Categorical levels for outcome Y; default: 1:4
- `Zlevels::AbstractRange`: Categorical levels for outcome Z; default: 1:3

# Returns
- `Tuple{Vector{Int}, Vector{Int}}`: Predicted outcomes (YB, ZA)
  - `YB`: Final predictions for Y in base B (argmax of network outputs)
  - `ZA`: Final predictions for Z in base A (argmax of network outputs)

# Algorithm
1. Initialize outcome prediction networks: modelXYA (X → Y), modelXZB (X → Z)
2. Compute cost matrix C based on Euclidean distance in **covariate space only**
3. Solve OT problem once: G = argmin ⟨G, C⟩ subject to marginal constraints
4. Transport outcome distributions (not covariates):
   - YBt = nB·YA·G (transported Y probabilities for base B samples)
   - ZAt = nA·ZB·G' (transported Z probabilities for base A samples)
5. Train networks on transported outcome targets:
   - Train modelXYA on (XB, YBt) - predict transported Y from base B covariates
   - Train modelXZB on (XA, ZAt) - predict transported Z from base A covariates
6. Return final predictions on original covariates

# Key Differences from Other DA Variants
- **da_covariables**: OT on X only; networks train on transported covariates (not outcomes)
- **da_outcomes**: OT on joint (X,Y) and (X,Z); iterative refinement (not implemented here)
- **da_outcomes_with_predictors** (this): OT on X only; networks train on transported **outcomes**
- **joint_ot_between_bases_with_predictors**: Iterative BCD with loss-driven cost updates

# Details
- **Cost matrix**: Based purely on covariate distances (SqEuclidean), normalized
- **Single OT solve**: Unlike BCD methods, OT problem solved once with static cost
- **Outcome transport**: Coupling G transports outcome probability distributions across bases
- **Network targets**: Soft probabilities (one-hot encoded) from transported distributions
- **Unbalanced OT**: Uses KL divergence for regularization when reg > 0
- **Prediction**: Uses softmax probabilities, then argmax for final discrete predictions

# See Also
- `joint_ot_between_bases_da_outcomes`: Similar approach (outcomes) without explicit predictors
- `joint_ot_between_bases_da_covariables`: OT on covariates with covariate-based DA
- `joint_ot_between_bases_with_predictors`: Iterative BCD variant with cost refinement
- `joint_ot_between_bases_category`: OT without discriminant analysis component

# Notes
- More computationally efficient than iterative BCD methods (single OT solve)
- Blends OT matching (for covariate alignment) with neural network flexibility (outcome prediction)
- Network training targets are transported outcome distributions, not standard supervised learning
- Outcome transport via coupling ensures covariate-aligned outcome distributions
- Useful when outcome prediction needs flexibility beyond linear transport assumptions
"""
function joint_ot_between_bases_da_outcomes_with_predictors(
        data;
        iterations = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 500,
        hidden_layer_size = 10,
        reg = 0.0,
        reg_m1 = 0.0,
        reg_m2 = 0.0,
        Ylevels = 1:4,
        Zlevels = 1:3
    )

    T = Int32

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    cols = names(dba, r"^X")

    XA = transpose(Matrix{Float32}(dba[:, cols]))
    XB = transpose(Matrix{Float32}(dbb[:, cols]))

    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)

    XYA = vcat(XA, YA)
    XZB = vcat(XB, ZB)

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(Float32, nA) ./ nA
    wb = ones(Float32, nB) ./ nB

    C0 = Float32.(pairwise(SqEuclidean(), XA, XB, dims = 2))
    C = C0 ./ maximum(C0)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelXYA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelXZB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))

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

    function loss_crossentropy(Y, F)

        ϵ = 1.0e-12
        res = zeros(Float32, size(Y, 2), size(F, 2))
        logF = zeros(Float32, size(F))

        for i in eachindex(F)
            if F[i] ≈ 1.0
                logF[i] = log(1.0 - ϵ)
            else
                logF[i] = log(ϵ)
            end
        end

        for i in axes(Y, 1)
            res .+= -Y[i, :] .* logF[i, :]'
        end

        return res

    end

    YBpred = Flux.softmax(modelXYA(XB))
    ZApred = Flux.softmax(modelXZB(XA))

    G = ones(Float32, length(wa), length(wb))

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end

    YBt = similar(YBpred)
    ZAt = similar(ZApred)

    YBt .= nB .* YA * G
    ZAt .= nA .* ZB * G'

    train!(modelXYA, XB, YBt)
    train!(modelXZB, XA, ZAt)

    YBpred .= Flux.softmax(modelXYA(XB))
    ZApred .= Flux.softmax(modelXZB(XA))

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

"""
$(SIGNATURES)

Optimal transport matching with outcome prediction from covariates only.

Solves OT problem on covariate space with single transport coupling G, then trains neural network
predictors to map covariates to outcomes. Networks learn to predict missing outcomes directly from
covariates (X → Y, X → Z) without using outcome combinations in the OT matching. This "without 
outcomes" variant focuses on covariate-outcome mapping rather than joint distribution transport.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
- `iterations::Int`: Number of BCD iterations (OT + network training cycles); default: 10
- `learning_rate::Float64`: Adam optimizer learning rate for networks; default: 0.01
- `batchsize::Int`: Batch size for stochastic gradient descent; default: 512
- `epochs::Int`: Training epochs per network at each BCD iteration; default: 1000
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

# Algorithm (Block Coordinate Descent)
1. Initialize single transport plan G and covariate-only predictor networks
2. For each BCD iteration:
   a. Solve OT problem on covariate space only:
      - G = argmin ⟨G, C⟩ (no outcome information in cost)
   b. Transport outcome distributions using single coupling:
      - YB = nB·YA·G (Y transported for base B)
      - ZA = nA·ZB·G' (Z transported for base A)
   c. Train predictor networks from covariates to transported outcomes:
      - Train modelXA on (XA, ZA) - learn X → Z
      - Train modelXB on (XB, YB) - learn X → Y
   d. Compute prediction errors (cross-entropy):
      - loss_y = YA vs modelXB(XB)
      - loss_z = ZB vs modelXA(XA)
   e. Update cost matrix with prediction error feedback:
      - C ← C₀ + loss_y + loss_z'
   f. Check convergence: transport plan stability or cost stability
3. Return argmax of final network predictions

# Key Differences from Similar Methods
- **without_outcomes** (this): OT on X only; networks predict X → Y, X → Z
- **with_predictors**: OT on X; networks train on transported (X,outcome) pairs
- **jdot**: Two separate OT problems (one per outcome); outcome-specific matching
- **da_outcomes**: OT on X; direct outcome distribution transport (no networks)

# Details
- **Covariate-only OT**: Cost matrix depends only on X distance, ignoring outcomes
- **Symmetric treatment**: Both Y and Z use same transport coupling G
- **Loss-driven refinement**: Prediction errors guide next OT cost matrix
- **Network inputs**: Covariate-only (X), not joint (X,outcome)
- **Transport targets**: Networks learn to match transported outcome distributions
- **Unbalanced OT**: Uses KL divergence for regularization when reg > 0
- **Single coupling**: Unlike JDOT (2 couplings), uses one G for both outcomes

# See Also
- `joint_ot_between_bases_with_predictors`: Joint (X,outcome) OT with predictors
- `joint_ot_between_bases_jdot`: Two separate OT problems (outcome-specific)
- `joint_ot_between_bases_da_covariables`: DA on covariates (no outcome info)
- `JointOTBetweenBases`: Main method dispatcher

# Notes
- Simplest BCD variant: matches on covariates, learns outcome relationships via networks
- Computationally efficient: single OT solve per iteration (vs 2 for JDOT)
- Outcome-agnostic matching: OT doesn't see outcome relationships directly
- Two networks learn identical relationship (X → Y and X → Z) from different transported targets
- Useful when outcome dimensions are complex or when covariate alignment is priority
"""
function joint_ot_between_bases_without_outcomes(
        data;
        iterations = 10,
        learning_rate = 0.01,
        batchsize = 512,
        epochs = 1000,
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
    C0 .= C0 ./ maximum(C0)
    C = copy(C0)

    dimXA = size(XA, 1)
    dimXB = size(XB, 1)

    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelXA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

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
            res .+= - view(Y, i, :) .* view(logF, i, :)'
        end

        return res

    end

    YBpred = Flux.softmax(modelXB(XB))
    ZApred = Flux.softmax(modelXA(XA))

    alpha1, alpha2 = 1 / length(Ylevels), 1 / length(Zlevels)

    G = ones(Float32, nA, nB)
    Gold = copy(G)
    cost = Inf

    YB = Float32.(nB .* YA * G)
    ZA = Float32.(nA .* ZB * G')

    for iter in 1:iterations # BCD algorithm

        costold = cost

        if reg > 0
            G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
        else
            G .= PythonOT.emd(wa, wb, C)
        end

        delta = norm(G .- Gold)
        Gold .= G

        YB .= nB .* YA * G
        ZA .= nA .* ZB * G'

        train!(modelXA, XA, ZA)
        train!(modelXB, XB, YB)

        YBpred .= Flux.softmax(modelXB(XB))
        ZApred .= Flux.softmax(modelXA(XA))

        loss_y = alpha1 * loss_crossentropy(YA, YBpred)
        loss_z = alpha2 * loss_crossentropy(ZB, ZApred)

        fcost = loss_y .+ loss_z'

        cost = sum(G .* fcost)

        @info "Delta: $(delta) \t  Loss: $(cost) "

        if delta < 1.0e-16 || abs(costold - cost) < 1.0e-7
            @info "converged at iter $iter "
            break
        end

        C .= C0 .+ fcost

    end

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

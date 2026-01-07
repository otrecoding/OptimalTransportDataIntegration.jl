"""
$(SIGNATURES)

Optimal transport matching with discriminant analysis on covariates.

Combines optimal transport on covariate space with neural network discriminant classifiers.
Solves OT problem based on covariate-only distance, then trains neural networks to predict
outcomes from transported covariates. The "DA" (Discriminant Analysis) prefix indicates that
prediction uses only covariates (X), not the joint (X,outcome) space like other DA variants.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
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
1. Initialize networks for outcome prediction: modelXYA (X → Y), modelXZB (X → Z)
2. Compute cost matrix C based on Euclidean distance in **covariate space only** (not outcomes)
3. Solve OT problem: find coupling G that minimizes ⟨G, C⟩
4. Transport covariates: XA_transported = nB·XA·G, XB_transported = nA·XB·G'
5. Train networks on transported covariates:
   - Train modelXYA on (XB_transported, YA)
   - Train modelXZB on (XA_transported, ZB)
6. Return final predictions on original covariates

# Key Differences from Other DA Variants
- **da_covariables** (this): OT on X only; predict Y,Z from X (covariate-only discriminant analysis)
- **da_outcomes**: OT on joint (X,Y) and (X,Z); may refine based on outcome prediction errors
- **da_outcomes_with_predictors**: Similar to da_outcomes but with explicit predictor networks

# Details
- **Cost matrix**: Based purely on covariate distances (SqEuclidean)
- **Network training**: Uses transported covariates as targets to match covariate distributions
- **Transport plan**: Couples full samples (not disaggregated by outcome), only balances marginal covariate distributions
- **Unbalanced OT**: Uses KL divergence for regularization when reg > 0

# See Also
- `joint_ot_between_bases_da_outcomes`: OT on outcomes with discriminant prediction
- `joint_ot_between_bases_da_outcomes_with_predictors`: Hybrid with explicit predictor networks
- `joint_ot_between_bases_category`: OT without discriminant analysis
- `joint_ot_between_bases_with_predictors`: OT with flexible predictor networks

# Notes
- Computationally less expensive than outcome-based DA variants
- May lose information by ignoring outcome distributions in OT matching
- Network training uses transported covariate targets, not standard supervised learning
- Useful when covariate matching is priority and outcome prediction is secondary
"""
function joint_ot_between_bases_da_covariables(
        data;
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

    C2 = pairwise(SqEuclidean(), XA, XB, dims = 2)
    C = C2 ./ maximum(C2)

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

    ZApred = modelXZB(XA)
    YBpred = modelXYA(XB)

    G = ones(Float32, nA, nB)

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end

    XAt = similar(XA)
    XBt = similar(XB)

    XBt .= nB .* XA * G
    XAt .= nA .* XB * G'

    train!(modelXYA, XAt, YA)
    train!(modelXZB, XBt, ZB)

    ZApred .= modelXZB(XA)
    YBpred .= modelXYA(XB)

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

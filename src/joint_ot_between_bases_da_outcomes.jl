"""
$(SIGNATURES)

Discriminant analysis via optimal transport on covariate-outcome space.

Solves a single optimal transport problem to match distributions across bases, then uses the
resulting transport coupling to predict outcomes. Unlike iterative refinement methods, this
approach computes outcomes directly from transported outcome distributions without predictor
networks. The "DA outcomes" variant focuses on transporting outcome probability distributions
aligned with covariate matching.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
- `iterations::Int`: Unused parameter (kept for API compatibility); default: 10
- `learning_rate::Float64`: Unused parameter (kept for API compatibility); default: 0.01
- `batchsize::Int`: Unused parameter (kept for API compatibility); default: 512
- `epochs::Int`: Unused parameter (kept for API compatibility); default: 500
- `hidden_layer_size::Int`: Unused parameter (kept for API compatibility); default: 10
- `reg::Float64`: Entropy regularization for OT (0 = exact, larger = relaxed); default: 0.0
- `reg_m1::Float64`: Marginal relaxation for base A; default: 0.0
- `reg_m2::Float64`: Marginal relaxation for base B; default: 0.0
- `Ylevels::AbstractRange`: Categorical levels for outcome Y; default: 1:4
- `Zlevels::AbstractRange`: Categorical levels for outcome Z; default: 1:3

# Returns
- `Tuple{Vector{Int}, Vector{Int}}`: Predicted outcomes (YB, ZA)
  - `YB`: Final predictions for Y in base B (argmax of transported outcome probabilities)
  - `ZA`: Final predictions for Z in base A (argmax of transported outcome probabilities)

# Algorithm
1. Compute cost matrix C based on Euclidean distance between covariates (X only)
2. Initialize uniform weights: wa = 1/nA, wb = 1/nB
3. Solve OT problem once: G = argmin ⟨G, C⟩ subject to marginal constraints
   - Uses unbalanced OT (KL divergence) if reg > 0, else exact EMD
4. Transport outcome distributions using coupling G:
   - YBpred = softmax(nB·YA·G): transported Y probabilities for base B samples
   - ZApred = softmax(nA·ZB·G'): transported Z probabilities for base A samples
5. Return argmax of transported probabilities as final predictions

# Key Differences from Other DA Variants
- **da_covariables**: OT on X only; predicts outcomes from transported covariates via networks
- **da_outcomes** (this): OT on X; predicts outcomes directly from transported outcome distributions
- **da_outcomes_with_predictors**: OT on X; trains networks on transported outcome targets
- **joint_ot_between_bases_with_predictors**: Iterative BCD with loss-driven cost updates

# Details
- **Cost matrix**: Based purely on covariate distances (SqEuclidean), not normalized
- **Single OT solve**: Static cost matrix, solved once (no iterative refinement)
- **Outcome transport**: Coupling G transports outcome probability distributions across bases
- **Direct prediction**: Predictions come directly from transported probabilities (softmax + argmax)
- **No networks**: Unlike with_predictors variant, uses no neural networks
- **Unbalanced OT**: Uses KL divergence for regularization when reg > 0
- **Scaling factor**: Transportation weighted by sample count (nB, nA) for proper probability aggregation

# See Also
- `joint_ot_between_bases_da_outcomes_with_predictors`: DA outcomes with explicit predictor networks
- `joint_ot_between_bases_da_covariables`: DA on covariates with covariate-based prediction
- `joint_ot_between_bases_category`: OT without discriminant analysis
- `joint_ot_between_bases_with_predictors`: Iterative BCD variant with cost refinement

# Notes
- Simplest DA variant: combines OT matching with direct outcome transport
- Most computationally efficient: single OT solve, no network training
- Outcome predictions are soft probabilities from transported distributions
- Useful for understanding how outcome distributions align under OT matching
- No learnable parameters; purely distribution-based prediction
- Hyperparameters for networks (learning_rate, etc.) are ignored for API compatibility
"""
function joint_ot_between_bases_da_outcomes(
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

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(nA) ./ nA
    wb = ones(nB) ./ nB

    C = Float32.(pairwise(SqEuclidean(), XA, XB, dims = 2))
    G = ones(Float32, length(wa), length(wb))

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end

    ZApred = Flux.softmax(nA .* ZB * G')
    YBpred = Flux.softmax(nB .* YA * G)

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

"""
    joint_ot_between_bases_jdot(data; iterations=10, learning_rate=0.01, batchsize=512, 
                                epochs=500, hidden_layer_size=10, reg=0.0, reg_m1=0.0, 
                                reg_m2=0.0, Ylevels=1:4, Zlevels=1:3)

Joint Distribution Optimal Transport (JDOT) with neural network predictors.

Iteratively solves separate OT problems for outcome Y and Z, each matching covariate distributions
while minimizing prediction error from outcome-specific discriminant classifiers. This block 
coordinate descent algorithm jointly optimizes two transport couplings (G1, G2) and two outcome 
prediction networks, enabling outcome-specific matching that balances covariate and outcome 
alignment.

The "JDOT" acronym stands for **Joint Distribution Optimal Transport**, reflecting the method's
focus on matching both covariate and outcome distributions across bases through independent 
transport plans per outcome.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

# Keyword Arguments
- `iterations::Int`: Number of BCD iterations (OT + network training cycles); default: 10
- `learning_rate::Float64`: Adam optimizer learning rate for networks; default: 0.01
- `batchsize::Int`: Batch size for stochastic gradient descent; default: 512
- `epochs::Int`: Training epochs per network at each BCD iteration; default: 500
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
1. Initialize two transport couplings G1, G2 and outcome prediction networks
2. For each BCD iteration:
   a. Solve two OT problems (one per outcome):
      - G1 = argmin ⟨G, C1⟩ matching for Y prediction errors
      - G2 = argmin ⟨G, C2⟩ matching for Z prediction errors
   b. Transport outcomes using outcome-specific couplings:
      - YB = nB·YA·G1 (Y transported for base B)
      - ZA = nA·ZB·G2' (Z transported for base A)
   c. Train discriminant networks on transported outcomes:
      - Train modelXYA on (XB, YB) for Y prediction
      - Train modelXZB on (XA, ZA) for Z prediction
   d. Compute prediction errors (cross-entropy):
      - loss_y = YA vs modelXYA(XB)
      - loss_z = ZB vs modelXZB(XA)
   e. Update cost matrices with prediction error feedback:
      - C1 ← C0 + loss_y
      - C2 ← C0 + loss_z'
   f. Check convergence: transport plan stability or cost stability
3. Return argmax of final network predictions

# Key Innovations of JDOT
- **Separate couplings per outcome**: G1 for Y matching, G2 for Z matching
- **Outcome-specific cost**: Each outcome has its own cost matrix driven by prediction errors
- **Joint optimization**: Balances covariate matching and outcome prediction accuracy
- **Iterative refinement**: Feedback loop where prediction errors guide next OT solve

# Differences from Related Methods
- **joint_ot_between_bases_with_predictors**: Single transport plan G; symmetric outcome treatment
- **joint_ot_between_bases_jdot** (this): Two transport plans G1, G2; outcome-specific matching
- **da_outcomes_with_predictors**: Single static OT solve; no cost refinement
- **joint_ot_between_bases_category**: No discriminant networks; pure OT on outcomes

# Details
- **Separate OT problems**: Each outcome solved independently, capturing outcome-specific distributions
- **Cost matrix updates**: Prediction errors guide next OT solve for better outcome alignment
- **Cross-entropy loss**: Weights outcome prediction errors by 1/n_levels for fair comparison
- **Transport scaling**: YB and ZA scaled by sample counts for proper probability aggregation
- **Discriminant classifiers**: Neural networks predict outcomes from covariates on transported samples
- **Unbalanced OT**: Uses KL divergence for regularization when reg > 0

# See Also
- `joint_ot_between_bases_with_predictors`: Single coupling variant
- `joint_ot_between_bases_da_outcomes_with_predictors`: DA variant (single static OT)
- `joint_ot_between_bases_category`: OT without discriminant networks
- `JointOTBetweenBases`: Main method dispatcher

# Notes
- Most sophisticated iterative approach: two independent transport problems per iteration
- Computationally expensive: ~2× cost of single-coupling methods per iteration
- Outcome-specific matching allows capturing different covariate-outcome relationships per outcome
- Useful when Y and Z have fundamentally different relationships to X
- Convergence typically faster than symmetric methods due to outcome-guided matching
"""
function joint_ot_between_bases_jdot(
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

    C0 = pairwise(SqEuclidean(), XA, XB, dims = 2)
    C1 = C0 ./ maximum(C0)
    C2 = C0 ./ maximum(C0)

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

    YBpred = modelXYA(XB)
    ZApred = modelXZB(XA)

    alpha1, alpha2 = 1 / length(Ylevels), 1 / length(Zlevels)

    G1 = ones(Float32, nA, nB)
    G2 = ones(Float32, nA, nB)
    cost = Inf

    YB = zeros(Float32, size(YA, 1), nB)
    ZA = zeros(Float32, size(ZB, 1), nA)

    loss_y = alpha1 .* loss_crossentropy(YA, YBpred)
    loss_z = alpha2 .* loss_crossentropy(ZB, ZApred)

    for iter in 1:iterations # BCD algorithm

        Gold = copy(G1)
        costold = cost

        if reg > 0.0
            G1 .= PythonOT.mm_unbalanced(wa, wb, C1, (reg_m1, reg_m2); reg = reg, div = "kl")
            G2 .= PythonOT.mm_unbalanced(wa, wb, C2, (reg_m1, reg_m2); reg = reg, div = "kl")
        else
            G1 .= PythonOT.emd(wa, wb, C1)
            G2 .= PythonOT.emd(wa, wb, C2)
        end

        delta = norm(G1 .- Gold)

        YB .= nB .* YA * G1
        ZA .= nA .* ZB * G2'

        train!(modelXYA, XB, YB)
        train!(modelXZB, XA, ZA)

        YBpred .= modelXYA(XB)
        ZApred .= modelXZB(XA)

        loss_y .= alpha1 .* loss_crossentropy(YA, YBpred)
        loss_z .= alpha2 .* loss_crossentropy(ZB, ZApred)

        fcost = loss_y .+ loss_z'

        cost = sum(G1 .* fcost)

        @info "Delta: $(delta) \t  Loss: $(cost) "

        if delta < 1.0e-16 || abs(costold - cost) < 1.0e-7
            @info "converged at iter $iter "
            break
        end

        C1 .= C0 ./ maximum(C0) .+ loss_y
        C2 .= C0 ./ maximum(C0) .+ loss_z'

    end

    return Flux.onecold(YBpred), Flux.onecold(ZApred)

end

"""
$(SIGNATURES)

Hybrid statistical matching combining optimal transport and neural network predictors.

Alternates between two operations: (1) solving optimal transport problem to match 
base A and B samples by covariate-outcome combinations, and (2) training neural networks 
to learn outcome prediction functions that minimize cross-entropy loss given transport plan.
This block coordinate descent (BCD) algorithm jointly optimizes coupling and predictors.

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
1. Initialize transport plan G, cost matrix C, and weights (uniform)
2. Concatenate covariates with outcomes: XYA = [XA; YA], XZB = [XB; ZB]
3. For each BCD iteration:
   - Solve OT problem to compute coupling G minimizing C
   - Update outcome targets: YB = nB·YA·G, ZA = nA·ZB·G'
   - Train predictor networks on updated targets
   - Compute cross-entropy loss and update cost matrix C ← C₀ + loss_feedback
   - Check convergence: transport plan stability (delta) or cost stability
4. Return argmax of final network predictions

# Details
- **Transport plan**: Couples samples across bases; G[i,j] represents mass transported from A[i] to B[j]
- **Predictors**: Neural networks learn mapping (X,outcome) → opposite_outcome with loss feedback
- **Cost update**: Cross-entropy loss between outcomes and network predictions guides next OT iteration
- **Unbalanced OT**: Uses KL divergence for regularization when reg > 0

# See Also
- `joint_ot_between_bases_category`: OT without neural network predictors
- `simple_learning`: Supervised baseline without OT coupling
- `JointOTBetweenBases`: Main method dispatcher

# Notes
- Computationally expensive: trains two networks at each of ~10 BCD iterations
- Combines strength of OT (covariate matching) with flexibility of neural networks
- Requires more iterations/epochs than pure OT for convergence
"""
function joint_ot_between_bases_with_predictors(
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

    colnames = names(data, r"^X")
    XA = transpose(Matrix{Float32}(dba[!, colnames]))
    XB = transpose(Matrix{Float32}(dbb[!, colnames]))

    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)

    XYA = vcat(XA, YA)
    XZB = vcat(XB, ZB)

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(Float32, nA) ./ nA
    wb = ones(Float32, nB) ./ nB

    C0 = pairwise(SqEuclidean(), XA, XB, dims = 2)
    C0 .= C0 ./ maximum(C0)

    C = copy(C0)

    dimXYA = size(XYA, 1)
    dimXZB = size(XZB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)

    modelXYA = Chain(Dense(dimXYA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXZB = Chain(Dense(dimXZB, hidden_layer_size), Dense(hidden_layer_size, dimYA))

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

    YBpred = modelXZB(XZB)
    ZApred = modelXYA(XYA)

    alpha1, alpha2 = 1 / length(Ylevels), 1 / length(Zlevels)

    G = ones(Float32, nA, nB)
    cost = Inf

    YB = nB .* YA * G
    ZA = nA .* ZB * G'

    for iter in 1:iterations # BCD algorithm

        Gold = copy(G)
        costold = cost

        if reg > 0
            G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
        else
            G .= PythonOT.emd(wa, wb, C)
        end

        delta = norm(G .- Gold)

        YB .= nB .* YA * G
        ZA .= nA .* ZB * G'

        train!(modelXYA, XYA, ZA)
        train!(modelXZB, XZB, YB)

        YBpred .= modelXZB(XZB)
        ZApred .= modelXYA(XYA)

        loss_y = alpha1 * loss_crossentropy(YA, YBpred)
        loss_z = alpha2 * loss_crossentropy(ZB, ZApred)

        fcost = loss_y .^ 2 .+ loss_z' .^ 2

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

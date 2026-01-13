using CSV
using DataFrames
import PythonOT
import .Iterators: product
import Distances: pairwise
import LinearAlgebra: norm


"""
$(SIGNATURES)

Statistical matching via optimal transport for discrete (categorical) covariates.

Specialized implementation for discrete covariate data. Aggregates individuals by unique covariate-outcome
combinations to reduce computational burden. Uses one-hot encoded covariates and solves OT problems on the
aggregated space. Iteratively minimizes cross-entropy loss between predicted and transported outcomes to
guide cost matrix refinement.

# Arguments
- `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
  `X*` covariates (must be categorical/integer), `Y` (outcome for base B), and `Z` (outcome for base A)
- `reg::Float64`: Entropy regularization parameter for OT solver
- `reg_m1::Float64`: Marginal constraint relaxation for base A
- `reg_m2::Float64`: Marginal constraint relaxation for base B

# Keyword Arguments
- `Ylevels::AbstractRange`: Categorical levels for outcome Y; default: 1:4
- `Zlevels::AbstractRange`: Categorical levels for outcome Z; default: 1:3
- `iterations::Int`: Number of algorithm iterations; default: 1
- `distance::Distances.Metric`: Distance metric for covariate space; default: Euclidean()

# Returns
- `Tuple{Vector{Int}, Vector{Int}}`: Predicted outcomes (YB, ZA)
  - `YB`: Predictions for Y in base B
  - `ZA`: Predictions for Z in base A

# Algorithm
1. One-hot encode discrete covariates
2. Aggregate individuals by unique (X, outcome) combinations:
   - Reduces individuals to "cells" (covariate × outcome modality combinations)
   - Computes cell weights as proportions of total sample size
   - Filters out empty cells
3. Compute pairwise distances between covariate profiles in aggregated space
4. Initialize outcome predictions and loss matrices
5. For each iteration:
   - Solve OT problem on aggregated cells with current cost matrix
   - Predict outcomes by minimizing cross-entropy loss (argmin over modalities)
   - Compute prediction error (cross-entropy between transported and predicted outcomes)
   - Update cost matrix with loss feedback
   - Check convergence
6. Map predictions back to original individual-level data
7. Return individual-level outcome predictions

# Computational Efficiency Tricks
- **Aggregation**: Reduces from n individuals to ~(n_covariates × n_outcomes) cells
  - E.g., 1000 individuals with 3 covariates (2,3,4 levels) + 2 outcomes → ~200-300 cells
- **One-hot encoding**: Discrete covariates represented as binary vectors for distance computation
- **Filtered weights**: Only processes cells with positive weight (observed data combinations)
- **Instance pre-computation**: Distance matrix computed once between all covariate profiles

# Details
- **Data aggregation**: Crucial for computational tractability with discrete data
- **Distance metric**: Determines covariate similarity (Euclidean on one-hot vectors by default)
- **Cost matrix**: Initial distance-based, updated with loss feedback
- **Cross-entropy**: Compares one-hot encoded outcomes for alignment
- **Convergence**: Checks transport plan stability (delta) and cost stability

# See Also
- `joint_ot_between_bases_category`: Continuous covariate version with categorical outcomes
- `Instance`: Pre-computed distance structure used in aggregation

# Notes
- **Discrete specialization**: Exploits discrete structure via aggregation for efficiency
- **Cell-level matching**: Matches entire cells (all individuals with same X and outcome)
- **Cross-entropy loss**: Drives iterative cost refinement for outcome prediction accuracy
- **Outcome aggregation**: Predictions map back from cell-level to individual-level deterministically
"""
function joint_ot_between_bases_discrete(
        data,
        reg,
        reg_m1,
        reg_m2;
        Ylevels = 1:4,
        Zlevels = 1:3,
        iterations = 1,
        distance = Euclidean(),
    )

    T = Int32

    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    colnames = names(data, r"^X")
    X_hot = Matrix{T}(one_hot_encoder(data[!, colnames]))
    Y = Vector{T}(data.Y)
    Z = Vector{T}(data.Z)

    YA = view(Y, indA)
    YB = view(Y, indB)
    ZA = view(Z, indA)
    ZB = view(Z, indB)

    XA = view(X_hot, indA, :)
    XB = view(X_hot, indB, :)

    YA_hot = one_hot_encoder(YA, Ylevels)
    ZA_hot = one_hot_encoder(ZA, Zlevels)
    YB_hot = one_hot_encoder(YB, Ylevels)
    ZB_hot = one_hot_encoder(ZB, Zlevels)

    XYA = hcat(XA, YA)
    XZB = hcat(XB, ZB)


    # Compute data for aggregation of the individuals

    nA = length(indA)
    nB = length(indB)

    # list the distinct modalities in A and B
    indY = Dict((m, findall(YA .== m)) for m in Ylevels)
    indZ = Dict((m, findall(ZB .== m)) for m in Zlevels)

    # Compute the indexes of individuals with same covariates
    indXA = Dict{T, Array{T}}()
    indXB = Dict{T, Array{T}}()
    Xlevels = sort(unique(eachrow(X_hot)))

    for (i, x) in enumerate(Xlevels)
        distA = vec(pairwise(distance, x[:, :], XA', dims = 2))
        distB = vec(pairwise(distance, x[:, :], XB', dims = 2))
        indXA[i] = findall(distA .< 0.1)
        indXB[i] = findall(distB .< 0.1)
    end

    nbXA = length(indXA)
    nbXB = length(indXB)

    wa = vec([length(indXA[x][findall(YA[indXA[x]] .== y)]) / nA for y in Ylevels, x in 1:nbXA])
    wb = vec([length(indXB[x][findall(ZB[indXB[x]] .== z)]) / nB for z in Zlevels, x in 1:nbXB])

    wa2 = filter(>(0), wa)
    wb2 = filter(>(0), wb)
    #wa = vec([sum(indXA[x][YA[indXA[x]] .== y]) for y in Ylevels, x in 1:nbXA])
    #wb = vec([sum(indXB[x][ZB[indXB[x]] .== z]) for z in Zlevels, x in 1:nbXB])
    #wa2 = filter(>(0), wa)
    #wb2 = filter(>(0), wb) ./ sum(wa2)

    XYA2 = Vector{T}[]
    XZB2 = Vector{T}[]
    i = 0
    for (y, x) in product(Ylevels, Xlevels)
        i += 1
        wa[i] > 0 && push!(XYA2, [x...; y])
    end
    i = 0
    for (z, x) in product(Zlevels, Xlevels)
        i += 1
        wb[i] > 0 && push!(XZB2, [x...; z])
    end

    # +
    Ylevels_hot = one_hot_encoder(Ylevels)
    Zlevels_hot = one_hot_encoder(Zlevels)

    nx = size(X_hot, 2) ## Nb modalités x

    # les x parmi les XYA observés, potentiellement des valeurs repetées
    XA_hot = stack([v[1:nx] for v in XYA2], dims = 1)
    # les x parmi les XZB observés, potentiellement des valeurs repetées
    XB_hot = stack([v[1:nx] for v in XZB2], dims = 1)

    yA = last.(XYA2) # les y parmi les XYA observés, potentiellement des valeurs repetées
    yA_hot = one_hot_encoder(yA, Ylevels)
    zB = last.(XZB2) # les z parmi les XZB observés, potentiellement des valeurs repetées
    zB_hot = one_hot_encoder(zB, Zlevels)

    # Algorithm

    ## Initialisation

    yB_pred = zeros(T, size(XZB2, 1)) # number of observed different values in A
    zA_pred = zeros(T, size(XYA2, 1)) # number of observed different values in B

    dimXZB = length(XZB2[1])
    dimXYA = length(XYA2[1])

    onecold(X) = map(argmax, eachrow(X))

    function loss_crossentropy(Y::Matrix{Bool}, F::Matrix{Bool})
        ϵ = 1.0e-12
        nf, nclasses = size(F)
        ny = size(Y, 1)
        @assert nclasses == size(Y, 2)
        res = zeros(Float32, ny, nf)
        logF = zeros(Float32, size(F))

        for i in eachindex(F)
            logF[i] = F[i] ≈ 1.0 ? log(1.0 - ϵ) : log(ϵ)
        end

        for i in axes(Y, 2)
            res .+= - view(Y, :, i) .* view(logF, :, i)'
        end

        return res

    end

    function modality_cost(loss::Matrix{Float32}, weight::Vector{Float32})

        cost_for_each_modality = Float64[]
        for j in axes(loss, 2)
            s = zero(Float64)
            for i in axes(loss, 1)
                s += loss[i, j] * weight[i]
            end
            push!(cost_for_each_modality, s)
        end

        return cost_for_each_modality

    end

    Yloss = loss_crossentropy(yA_hot, Ylevels_hot)
    Zloss = loss_crossentropy(zB_hot, Zlevels_hot)

    alpha1 = 1 / maximum(loss_crossentropy(Ylevels_hot, Ylevels_hot))
    alpha2 = 1 / maximum(loss_crossentropy(Zlevels_hot, Zlevels_hot))

    ## Optimal Transport

    C0 = pairwise(distance, XA_hot, XB_hot; dims = 1)

    C0 = C0 ./ maximum(C0)
    C0 .= C0 .^ 2
    C = C0
    zA_pred_hot_i = zeros(T, (nA, length(Zlevels)))
    yB_pred_hot_i = zeros(T, (nB, length(Ylevels)))

    est_opt = 0.0

    YBpred = zeros(T, nB)
    ZApred = zeros(T, nA)

    G = ones(Float32, length(wa2), length(wb2))
    cost = Inf

    for iter in 1:iterations

        Gold = copy(G)
        costold = cost

        if reg_m1 > 0.0 && reg_m2 > 0.0
            G .= PythonOT.mm_unbalanced(wa2, wb2, C, (reg_m1, reg_m2); reg = reg, div = "kl")
        else
            G .= PythonOT.emd(wa2, wb2, C)
        end

        delta = norm(G .- Gold)


        for j in eachindex(yB_pred)
            yB_pred[j] = Ylevels[argmin(modality_cost(Yloss, G[:, j]))]
        end

        for i in eachindex(zA_pred)
            zA_pred[i] = Zlevels[argmin(modality_cost(Zloss, G[i, :]))]
        end

        yB_pred_hot = one_hot_encoder(yB_pred, Ylevels)
        zA_pred_hot = one_hot_encoder(zA_pred, Zlevels)

        ### Update Cost matrix

        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)

        fcost = chinge1 .^ 2 .+ chinge2' .^ 2

        cost = sum(G .* fcost)

        @info "Delta: $(delta) \t  Loss: $(cost) "

        C .= C0 .+ fcost

        for i in axes(XYA, 1)
            ind = findfirst(XYA[i, :] == v for v in XYA2)
            zA_pred_hot_i[i, :] .= zA_pred_hot[ind, :]
        end

        for i in axes(XZB, 1)
            ind = findfirst(XZB[i, :] == v for v in XZB2)
            yB_pred_hot_i[i, :] .= yB_pred_hot[ind, :]
        end

        YBpred .= onecold(yB_pred_hot_i)
        ZApred .= onecold(zA_pred_hot_i)

        if delta < 1.0e-16 || abs(costold - cost) < 1.0e-7
            @info "converged at iter $iter "
            break
        end

    end

    return YBpred, ZApred


end

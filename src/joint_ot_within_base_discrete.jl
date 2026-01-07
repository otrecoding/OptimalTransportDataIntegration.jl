using JuMP, Clp

"""
$(SIGNATURES)

Solve linear program for within-base outcome balancing via optimal transport.

Core solver for within-base OT matching on discrete covariates. Takes pre-computed Instance
with distance matrices and individual aggregation, then formulates and solves linear programs
(one per base A, B) to find joint distributions of (X, Y, Z) that balance covariate and 
outcome distributions while minimizing transportation cost.

# Arguments
- `inst::Instance`: Pre-computed instance with distance matrices and aggregation indices
- `alpha::Float64`: Weight balancing covariate regularization (higher = prioritize covariates)
- `lambda::Float64`: Coefficient controlling overall regularization strength
- `percent_closest::Float64`: Fraction of closest neighbors for cost computation (robustness)

# Keyword Arguments
- `norme::Metric`: Distance metric for covariate neighbors; default: Euclidean()
- `aggregate_tol::Float64`: Tolerance for covariate aggregation (unused in main algorithm); default: 0.5
- `verbose::Bool`: Enable detailed logging of results; default: false

# Returns
- `Solution`: Object containing optimal transport plan and outcome predictions

# Algorithm Details
1. Pre-aggregate individuals by similar covariate values (using Instance)
2. Compute cost matrix C[y,z] from instance pre-computed distances
3. Build linear program per base with:
   - Variables: γ[x,y,z] joint probabilities of (covariate, Y, Z)
   - Constraints: marginals match observed distributions
   - Objective: minimize transport cost + regularization
4. Solve LP using Clp (simplex algorithm)
5. Extract solution and compute outcome predictions

# Model Structure
The LP formulates the problem:
```
min  ⟨C, γ⟩ + λ·(regularization on covariate matching)
s.t. marginal constraints ensuring statistical consistency
     γ ≥ 0 (probabilities non-negative)
```

# Optimization Targets
- **Covariate alignment**: Regularization ensures covariate marginals stay close to observed
- **Outcome prediction**: Cost matrix drives outcome rebalancing via OT
- **Stability**: Neighborhood averaging and regularization prevent overfitting

# See Also
- `Instance`: Pre-computation structure
- `average_distance_to_closest`: Cost matrix computation
- `Solution`: Output structure

# Notes
- Requires pre-computed Instance for efficiency
- Two independent LP solves (base A and base B)
- Uses Clp solver for exactness (not approximate)
"""
function ot_joint(
        inst::Instance,
        alpha::Float64,
        lambda::Float64,
        percent_closest::Float64;
        norme::Metric = Euclidean(),
        aggregate_tol::Float64 = 0.5,
        verbose::Bool = false,
    )

    if verbose
        @info " AGGREGATE INDIVIDUALS WRT COVARIATES               "
        @info " Reg. weight           = $(lambda)              "
        @info " Percent closest       = $(100.0 * percent_closest) % "
        @info " Aggregation tolerance = $(aggregate_tol)           "
    end

    # Local redefinitions of parameters of  the instance
    nA = inst.nA
    nB = inst.nB
    A = 1:nA
    B = 1:nB
    Ylevels = 1:4
    Zlevels = 1:3
    indY = inst.indY
    indZ = inst.indZ
    Xobserv = inst.Xobserv
    Yobserv = inst.Yobserv
    Zobserv = inst.Zobserv

    # Create a model for the optimal transport of individuals
    modelA = Model(Clp.Optimizer)
    modelB = Model(Clp.Optimizer)
    set_optimizer_attribute(modelA, "LogLevel", 0)
    set_optimizer_attribute(modelB, "LogLevel", 0)

    # Compute data for aggregation of the individuals
    # println("... aggregating individuals")
    indXA = inst.indXA
    indXB = inst.indXB
    Xlevels = eachindex(indXA)

    # compute the neighbors of the covariates for regularization
    Xvalues = unique(eachrow(Xobserv))
    dist_X = pairwise(norme, Xvalues, Xvalues)
    voisins = findall.(eachrow(dist_X .<= 1))
    nvoisins = length(Xvalues)

    # println("... computing costs")
    C = average_distance_to_closest(inst, percent_closest)

    # Compute the estimators that appear in the model

    estim_XA = length.(indXA) ./ nA
    estim_XB = length.(indXB) ./ nB
    estim_XA_YA = [
        length(indXA[x][findall(Yobserv[indXA[x]] .== y)]) / nA for
            x in Xlevels, y in Ylevels
    ]
    estim_XB_ZB = [
        length(indXB[x][findall(Zobserv[indXB[x] .+ nA] .== z)]) / nB for
            x in Xlevels, z in Zlevels
    ]


    # Basic part of the model

    # Variables
    # - gammaA[x,y,z]: joint probability of X=x, Y=y and Z=z in base A
    @variable(
        modelA,
        gammaA[x in Xlevels, y in Ylevels, z in Zlevels] >= 0,
        base_name = "gammaA"
    )

    # - gammaB[x,y,z]: joint probability of X=x, Y=y and Z=z in base B
    @variable(
        modelB,
        gammaB[x in Xlevels, y in Ylevels, z in Zlevels] >= 0,
        base_name = "gammaB"
    )

    @variable(modelA, errorA_XY[x in Xlevels, y in Ylevels], base_name = "errorA_XY")
    @variable(
        modelA,
        abserrorA_XY[x in Xlevels, y in Ylevels] >= 0,
        base_name = "abserrorA_XY"
    )
    @variable(modelA, errorA_XZ[x in Xlevels, z in Zlevels], base_name = "errorA_XZ")
    @variable(
        modelA,
        abserrorA_XZ[x in Xlevels, z in Zlevels] >= 0,
        base_name = "abserrorA_XZ"
    )

    @variable(modelB, errorB_XY[x in Xlevels, y in Ylevels], base_name = "errorB_XY")
    @variable(
        modelB,
        abserrorB_XY[x in Xlevels, y in Ylevels] >= 0,
        base_name = "abserrorB_XY"
    )
    @variable(modelB, errorB_XZ[x in Xlevels, z in Zlevels], base_name = "errorB_XZ")
    @variable(
        modelB,
        abserrorB_XZ[x in Xlevels, z in Zlevels] >= 0,
        base_name = "abserrorB_XZ"
    )

    # Constraints
    # - assign sufficient probability to each class of covariates with the same outcome
    @constraint(
        modelA,
        ctYandXinA[x in Xlevels, y in Ylevels],
        sum(gammaA[x, y, z] for z in Zlevels) == estim_XA_YA[x, y] + errorA_XY[x, y]
    )
    @constraint(
        modelB,
        ctZandXinB[x in Xlevels, z in Zlevels],
        sum(gammaB[x, y, z] for y in Ylevels) == estim_XB_ZB[x, z] + errorB_XZ[x, z]
    )

    # - we impose that the probability of Y conditional to X is the same in the two databases
    # - the consequence is that the probability of Y and Z conditional to Y is also the same in the two bases
    @constraint(
        modelA,
        ctZandXinA[x in Xlevels, z in Zlevels],
        estim_XB[x] * sum(gammaA[x, y, z] for y in Ylevels) ==
            estim_XB_ZB[x, z] * estim_XA[x] + estim_XB[x] * errorA_XZ[x, z]
    )

    @constraint(
        modelB,
        ctYandXinB[x in Xlevels, y in Ylevels],
        estim_XA[x] * sum(gammaB[x, y, z] for z in Zlevels) ==
            estim_XA_YA[x, y] * estim_XB[x] + estim_XA[x] * errorB_XY[x, y]
    )

    # - recover the norm 1 of the error
    @constraint(modelA, [x in Xlevels, y in Ylevels], errorA_XY[x, y] <= abserrorA_XY[x, y])
    @constraint(
        modelA,
        [x in Xlevels, y in Ylevels],
        -errorA_XY[x, y] <= abserrorA_XY[x, y]
    )
    @constraint(
        modelA,
        sum(abserrorA_XY[x, y] for x in Xlevels, y in Ylevels) <= alpha / 2.0
    )
    @constraint(modelA, sum(errorA_XY[x, y] for x in Xlevels, y in Ylevels) == 0.0)
    @constraint(modelA, [x in Xlevels, z in Zlevels], errorA_XZ[x, z] <= abserrorA_XZ[x, z])
    @constraint(
        modelA,
        [x in Xlevels, z in Zlevels],
        -errorA_XZ[x, z] <= abserrorA_XZ[x, z]
    )

    @constraint(
        modelA,
        sum(abserrorA_XZ[x, z] for x in Xlevels, z in Zlevels) <= alpha / 2.0
    )

    @constraint(modelA, sum(errorA_XZ[x, z] for x in Xlevels, z in Zlevels) == 0.0)

    @constraint(modelB, [x in Xlevels, y in Ylevels], errorB_XY[x, y] <= abserrorB_XY[x, y])

    @constraint(
        modelB,
        [x in Xlevels, y in Ylevels],
        -errorB_XY[x, y] <= abserrorB_XY[x, y]
    )

    @constraint(
        modelB,
        sum(abserrorB_XY[x, y] for x in Xlevels, y in Ylevels) <= alpha / 2.0
    )

    @constraint(modelB, sum(errorB_XY[x, y] for x in Xlevels, y in Ylevels) == 0.0)

    @constraint(modelB, [x in Xlevels, z in Zlevels], errorB_XZ[x, z] <= abserrorB_XZ[x, z])

    @constraint(
        modelB,
        [x in Xlevels, z in Zlevels],
        -errorB_XZ[x, z] <= abserrorB_XZ[x, z]
    )

    @constraint(
        modelB,
        sum(abserrorB_XZ[x, z] for x in Xlevels, z in Zlevels) <= alpha / 2.0
    )

    @constraint(modelB, sum(errorB_XZ[x, z] for x in Xlevels, z in Zlevels) == 0.0)

    # - regularization
    @variable(
        modelA,
        reg_absA[x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels] >= 0
    )

    @constraint(
        modelA,
        [x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels],
        reg_absA[x1, x2, y, z] >=
            gammaA[x1, y, z] / (max(1, length(indXA[x1])) / nA) -
            gammaA[x2, y, z] / (max(1, length(indXA[x2])) / nA)
    )

    @constraint(
        modelA,
        [x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels],
        reg_absA[x1, x2, y, z] >=
            gammaA[x2, y, z] / (max(1, length(indXA[x2])) / nA) -
            gammaA[x1, y, z] / (max(1, length(indXA[x1])) / nA)
    )

    @expression(
        modelA,
        regterm,
        sum(
            1 / nvoisins * reg_absA[x1, x2, y, z] for
                x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels
        )
    )

    @variable(
        modelB,
        reg_absB[x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels] >= 0
    )

    @constraint(
        modelB,
        [x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels],
        reg_absB[x1, x2, y, z] >=
            gammaB[x1, y, z] / (max(1, length(indXB[x1])) / nB) -
            gammaB[x2, y, z] / (max(1, length(indXB[x2])) / nB)
    )

    @constraint(
        modelB,
        [x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels],
        reg_absB[x1, x2, y, z] >=
            gammaB[x2, y, z] / (max(1, length(indXB[x2])) / nB) -
            gammaB[x1, y, z] / (max(1, length(indXB[x1])) / nB)
    )

    @expression(
        modelB,
        regterm,
        sum(
            1 / nvoisins * reg_absB[x1, x2, y, z] for
                x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels
        )
    )

    # by default, the OT cost and regularization term are weighted to lie in the same interval
    @objective(
        modelA,
        Min,
        sum(C[y, z] * gammaA[x, y, z] for y in Ylevels, z in Zlevels, x in Xlevels) +
            lambda * sum(
            1 / nvoisins * reg_absA[x1, x2, y, z] for
                x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels
        )
    )

    @objective(
        modelB,
        Min,
        sum(C[y, z] * gammaB[x, y, z] for y in Ylevels, z in Zlevels, x in Xlevels) +
            lambda * sum(
            1 / nvoisins * reg_absB[x1, x2, y, z] for
                x1 in Xlevels, x2 in voisins[x1], y in Ylevels, z in Zlevels
        )
    )

    # Solve the problem
    optimize!(modelA)
    optimize!(modelB)

    # Extract the values of the solution
    gammaA_val = [value(gammaA[x, y, z]) for x in Xlevels, y in Ylevels, z in Zlevels]
    gammaB_val = [value(gammaB[x, y, z]) for x in Xlevels, y in Ylevels, z in Zlevels]

    # compute the resulting estimators for the distributions of Z
    # conditional to X and Y in base A and of Y conditional to X and Z in base B

    estimatorZA = ones(length(Xlevels), length(Ylevels), length(Zlevels)) ./ length(Zlevels)

    for x in Xlevels, y in Ylevels
        proba_c_mA = sum(gammaA_val[x, y, Zlevels])
        if proba_c_mA > 1.0e-6
            estimatorZA[x, y, :] = gammaA_val[x, y, :] ./ proba_c_mA
        end
    end

    estimatorYB = ones(length(Xlevels), length(Ylevels), length(Zlevels)) ./ length(Ylevels)

    for x in Xlevels, z in Zlevels
        proba_c_mB = sum(view(gammaB_val, x, Ylevels, z))
        if proba_c_mB > 1.0e-6
            estimatorYB[x, :, z] = view(gammaB_val, x, :, z) ./ proba_c_mB
        end
    end

    if verbose
        solution_summary(modelA; verbose = verbose)
        solution_summary(modelB; verbose = verbose)
    end

    return Solution(
        [sum(gammaA_val[:, y, z]) for y in Ylevels, z in Zlevels],
        [sum(gammaB_val[:, y, z]) for y in Ylevels, z in Zlevels],
        estimatorZA,
        estimatorYB,
    )

end

function joint_ot_within_base_discrete(
        data;
        lambda = 0.392,
        alpha = 0.714,
        percent_closest = 0.2,
        distance = Euclidean(),
    )

    """
        joint_ot_within_base_discrete(data; lambda=0.392, alpha=0.714, percent_closest=0.2, distance=Euclidean())

    Balance within-base distributions via optimal transport for discrete covariates.

    Wrapper function for within-base OT on discrete covariate data. Builds Instance structure
    with pre-computed distances and aggregation, then solves linear programs independently for
    each base (A and B) to balance covariate and outcome distributions. Does NOT match across
    bases—focuses on internal distribution alignment within each data source.

    # Arguments
    - `data::DataFrame`: Input data with columns `database` (1 for base A, 2 for base B), 
      `X*` discrete covariates, `Y` (outcome for base B), and `Z` (outcome for base A)

    # Keyword Arguments
    - `lambda::Float64`: Regularization weight for covariate smoothness; default: 0.392
    - `alpha::Float64`: Balance weight between covariate and outcome terms; default: 0.714
    - `percent_closest::Float64`: Fraction of closest neighbors for cost robustness; default: 0.2
    - `distance::Distances.Metric`: Distance metric for covariate aggregation; default: Euclidean()

    # Returns
    - `Tuple{Vector{Int}, Vector{Int}}`: Predicted outcomes (YB, ZA)
      - `YB`: Balanced outcome predictions for base B
      - `ZA`: Balanced outcome predictions for base A

    # Algorithm
    1. Extract database, covariates, and outcomes from DataFrame
    2. Build Instance: pre-compute distances and individual aggregation by covariates
    3. Solve ot_joint LP solver:
       - Formulate linear program per base (A, B)
       - Optimize joint distribution γ[x,y,z] for (covariate, outcome pairs)
       - Minimize: transportation cost + regularization
    4. Extract outcome predictions from solution
    5. Return predicted outcomes

    # Key Differences from Between-Bases Methods
    - **Within-base**: Solves independent LP for A and B (no cross-base information)
    - **No integration**: Cannot leverage relationships between bases
    - **Within-base only**: Useful as reference baseline or when bases should stay separate
    - **Faster**: Independent problems smaller than joint optimization

    # Details
    - **Discrete specialization**: Aggregates individuals by identical covariate patterns
    - **Instance pre-computation**: Caches distances and aggregation for efficiency
    - **Regularization**: Controls balance between covariate alignment and outcome improvement
    - **LP exactness**: Clp solver guarantees optimal solution (not approximate)
    - **Default parameters**: Calibrated for typical statistical matching scenarios

    # See Also
    - `ot_joint`: Core LP solver
    - `Instance`: Pre-computation structure
    - `joint_ot_within_base_continuous`: Continuous covariate version
    - `JointOTBetweenBases`: Between-bases integration approach

    # Notes
    - Reference implementation: demonstrates within-base balancing capability
    - Limited matching: does not use cross-base information for outcome prediction
    - Aggregation efficiency: works well when covariates have discrete structure
    - Outcome levels: fixed to Y ∈ {1,2,3,4}, Z ∈ {1,2,3}
    """

    database = data.database

    Xnames = names(data, r"^X")

    X = Matrix(data[!, Xnames])
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    Ylevels = 1:4
    Zlevels = 1:3

    instance = Instance(database, X, Y, Ylevels, Z, Zlevels, distance)

    sol = ot_joint(instance, alpha, lambda, percent_closest)
    YB, ZA = compute_pred_error!(sol, instance, false)

    return YB, ZA

end

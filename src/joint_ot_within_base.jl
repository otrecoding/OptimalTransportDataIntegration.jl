using JuMP, Clp

"""
$(SIGNATURES)

Model where we directly compute the distribution of the outcomes for each
individual or for sets of indviduals that similar values of covariates

- aggregate_tol: quantify how much individuals' covariates must be close for aggregation
- reg_norm: norm1, norm2 or entropy depending on the type of regularization
- percent_closest: percent of closest neighbors taken into consideration in regularization
- lambda: coefficient measuring the importance of the regularization term
- verbose: if true, write the transported value of each individual; otherwise, juste write the number of missed transports
"""
function ot_joint(
        inst::Instance,
        alpha::Float64,
        lambda::Float64,
        percent_closest::Float64;
        norme::Metric = Cityblock(),
        aggregate_tol::Float64 = 0.5,
        verbose::Bool = false,
    )

    if verbose
        @info " AGGREGATE INDIVIDUALS WRT COVARIATES               "
        @info " Reg. weight           = $(lambda)              "
        @info " Percent closest       = $(100.0 * percent_closest) % "
        @info " Aggregation tolerance = $(aggregate_tol)           "
    end

    tstart = time()

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
        time() - tstart,
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
        distance = Hamming(),
    )

    database = data.database

    Xnames = ["X1", "X2", "X3"]

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


function joint_ot_within_base_continuous(
        data;
        lambda = 0.392,
        alpha = 0.714,
        percent_closest = 0.2,
        distance = Euclidean(),
    )

    digitize(x, bins) = searchsortedlast.(Ref(bins), x)

    XA = subset(data, :database => x -> x .== 1.0)
    XB = subset(data, :database => x -> x .== 2.0)

    b1 = quantile(data.X1, collect(0.1:0.1:0.9))
    bins11 = vcat(-Inf, b1, +Inf)

    X11 = digitize(XA.X1, bins11)
    X21 = digitize(XB.X1, bins11)

    b1 = quantile(data.X2, collect(0.1:0.1:0.9))
    bins12 = vcat(-Inf, b1, +Inf)

    X12 = digitize(XA.X2, bins12)
    X22 = digitize(XB.X2, bins12)

    b1 = quantile(data.X3, collect(0.1:0.1:0.9))
    bins13 = vcat(-Inf, b1, +Inf)

    X13 = digitize(XA.X3, bins13)
    X23 = digitize(XB.X3, bins13)

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)

    X = hcat(X1, X2, X3)
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    Ylevels = 1:4
    Zlevels = 1:3

    database = data.database

    instance = Instance(database, X, Y, Ylevels, Z, Zlevels, distance)

    sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

    return compute_pred_error!(sol, instance, false)

end

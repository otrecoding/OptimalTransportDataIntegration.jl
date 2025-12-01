import Statistics: median

function joint_ot_within_base_continuous(
        data;
        lambda = 0.392,
        alpha = 0.714,
        percent_closest = 0.2,
        distance = Euclidean(),
        Ylevels = 1:4,
        Zlevels = 1:3
    )

    digitize(x, bins) = searchsortedlast.(Ref(bins), x)

    XA = subset(data, :database => x -> x .== 1)
    XB = subset(data, :database => x -> x .== 2)

    X = Vector{Int}[]
    Xmdn = Vector{Float64}[]
    for col in names(data, r"^X")

        b = quantile(data[!, col], collect(0.25:0.25:0.75))
        bins = vcat(-Inf, b, +Inf)

        X1 = digitize(XA[!, col], bins)
        X2 = digitize(XB[!, col], bins)
        push!(X, vcat(X1, X2))

        X1mdn = zeros(Float32, size(X1, 1))
        for i in unique(X1)
            mdn = median(XA[X1 .== i, col])
            X1mdn[X1 .== i] .= mdn
        end

        X2mdn = zeros(Float32, size(X2, 1))
        for i in unique(X2)
            mdn = median(XB[X2 .== i, col])
            X2mdn[X2 .== i] .= mdn
        end

        push!(Xmdn, vcat(X1mdn, X2mdn))

    end
    X = stack(X)
    Xmdn = stack(Xmdn)
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    Xobserv = vcat(X[indA, :], X[indB, :])
    Yobserv = vcat(Y[indA], Y[indB])
    Zobserv = vcat(Z[indA], Z[indB])

    nA = length(indA)
    nB = length(indB)

    # list the distinct modalities in A and B
    indY = [findall(Y[indA] .== m) for m in Ylevels]
    indZ = [findall(Z[indB] .== m) for m in Zlevels]

    # compute the distance between pairs of individuals in different bases
    # devectorize all the computations to go about twice faster only compute norm 1 here

    a = Xmdn[indA, :]'
    b = Xmdn[indB, :]'

    D = pairwise(distance, a, b, dims = 2)

    # Compute the indexes of individuals with same covariates
    indXA = Vector{Int64}[]
    indXB = Vector{Int64}[]

    Xlevels = unique(eachrow(X))
    # aggregate both bases
    a = X[indA, :]'
    b = X[indB, :]'
    for x in Xlevels
        distA = vec(pairwise(distance, x[:, :], a, dims = 2))
        push!(indXA, findall(distA .< 0.1))
    end

    Xlevels = unique(eachrow(X))
    for x in Xlevels
        distB = vec(pairwise(distance, x[:, :], b, dims = 2))
        push!(indXB, findall(distB .< 0.1))
    end


    norme = Euclidean()
    aggregate_tol = 0.5

    # Create a model for the optimal transport of individuals
    modelA = Model(Clp.Optimizer)
    modelB = Model(Clp.Optimizer)
    set_optimizer_attribute(modelA, "LogLevel", 0)
    set_optimizer_attribute(modelB, "LogLevel", 0)

    # Compute data for aggregation of the individuals
    Xlevels = eachindex(indXA)

    # compute the neighbors of the covariates for regularization
    Xvalues = unique(eachrow(Xobserv))
    dist_X = pairwise(norme, Xvalues, Xvalues)
    voisins = findall.(eachrow(dist_X .<= 1))
    nvoisins = length(Xvalues)

    C = zeros(Float64, (length(Ylevels), length(Zlevels)))

    for y in Ylevels, i in indY[y], z in Zlevels

        nbclose = round(Int, percent_closest * length(indZ[z]))
        if nbclose > 0
            distance = [D[i, j] for j in indZ[z]]
            p = partialsortperm(distance, 1:nbclose)
            C[y, z] += sum(distance[p]) / nbclose / length(indY[y]) / 2.0
        end

    end

    for z in Zlevels, j in indZ[z], y in Ylevels

        nbclose = round(Int, percent_closest * length(indY[y]))
        if nbclose > 0
            distance = [D[i, j] for i in indY[y]]
            p = partialsortperm(distance, 1:nbclose)
            C[y, z] += sum(distance[p]) / nbclose / length(indZ[z]) / 2.0
        end

    end

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

    nbX = length(indXA)

    # Count the number of mistakes in the transport
    #deduce the individual distributions of probability for each individual from the distributions
    probaZindivA = zeros(Float64, (nA, length(Zlevels)))
    probaYindivB = zeros(Float64, (nB, length(Ylevels)))
    for x in 1:nbX
        for i in indXA[x]
            probaZindivA[i, :] .= estimatorZA[x, Yobserv[i], :]
        end
        for i in indXB[x]
            probaYindivB[i, :] .= estimatorYB[x, :, Zobserv[i + nA]]
        end
    end

    # Transport the modality that maximizes frequency
    predZA = [findmax([probaZindivA[i, z] for z in Zlevels])[2] for i in 1:nA]
    predYB = [findmax([probaYindivB[j, y] for y in Ylevels])[2] for j in 1:nB]

    return predYB, predZA

end

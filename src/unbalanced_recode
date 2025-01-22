 #CHANGE PythonOT.entropic_partial_wasserstein(wa2, wb2, C, reg,m)

# fonction qui dépend de max_relax et lambda_reg


@variable(
        modelA,
        gammaA[x1 in 1:nbX, y in Ylevels, x2 in 1:nbX,z in Zlevels] >= 0,
        base_name = "gammaA"
    )

      
    @variable(modelA, errorA_XY[x1 in 1:nbX, y in Ylevels], base_name = "errorA_XY")
    @variable(
        modelA,
        abserrorA_XY[x1 in 1:nbX, y in Ylevels] >= 0,
        base_name = "abserrorA_XY"
    )
    @variable(modelA, errorA_XZ[x2 in 1:nbX, z in Zlevels], base_name = "errorA_XZ")
    @variable(
        modelA,
        abserrorA_XZ[x2 in 1:nbX, z in Zlevels] >= 0,
        base_name = "abserrorA_XZ"
    )

     # Constraints
    # - assign sufficient probability to each class of covariates with the same outcome
    @constraint(
        modelA,
        ctYandXinA[x1 in 1:nbX, y in Ylevels],
        sum(gammaA[x1, y,x2, z] for z in Zlevels) == estim_XA_YA[x1, y] + errorA_XY[x1, y]
    )


    # - we impose that the probability of Y conditional to X is the same in the two databases
    # - the consequence is that the probability of Y and Z conditional to Y is also the same in the two bases
    @constraint(
        modelA,
        ctZandXinA[x2 in 1:nbX, z in Zlevels],
        sum(gammaA[x1, y,x2, z] for y in Ylevels) ==
        estim_XB_ZB[x2, z] + errorA_XZ[x2, z]
    )

  
    # - recover the norm 1 of the error
    @constraint(modelA, [x1 in 1:nbX, y in Ylevels], errorA_XY[x1, y] <= abserrorA_XY[x1, y])
    @constraint(modelA, [x1 in 1:nbX, y in Ylevels], -errorA_XY[x1, y] <= abserrorA_XY[x1, y])
    @constraint(
        modelA,
        sum(abserrorA_XY[x1, y] for x1 = 1:nbX, y in Ylevels) <= maxrelax / 2.0
    )
    @constraint(modelA, sum(errorA_XY[x1, y] for x1 = 1:nbX, y in Ylevels) == 0.0)
    @constraint(modelA, [x2 in 1:nbX, z in Zlevels], errorA_XZ[x2, z] <= abserrorA_XZ[x2, z])
    @constraint(modelA, [x2 in 1:nbX, z in Zlevels], -errorA_XZ[x2, z] <= abserrorA_XZ[x2, z])
    @constraint(
        modelA,
        sum(abserrorA_XZ[x2, z] for x2 = 1:nbX, z in Zlevels) <= maxrelax / 2.0
    )
    @constraint(modelA, sum(errorA_XZ[x2, z] for x2 = 1:nbX, z in Zlevels) == 0.0)

  
    # - regularization
    @variable(
        modelA,
        reg_absA[
            x1 in 1:nbX,
            x2 in findall(voisins_X[x1, :]),
            y in Ylevels,
            z in Zlevels,
        ] >= 0
    )
    @constraint(
        modelA,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), y in Ylevels, x in in 1:nbX, z in Zlevels],
        reg_absA[x1, x2, y, z] >=
        gammaA[x1, y, x,z] / (max(1, length(indXA[x1])) / nA) -
        gammaA[x2, y, x,z] / (max(1, length(indXA[x2])) / nA)
    )
    @constraint(
        modelA,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), y in Ylevels, x in in 1:nbX, z in Zlevels],
        reg_absA[x1, x2, y, z] >=
        gammaA[x2, y,x, z] / (max(1, length(indXA[x2])) / nA) -
        gammaA[x1, y, x,z] / (max(1, length(indXA[x1])) / nA)
    )
    @expression(
        modelA,
        regterm,
        sum(
            1 / length(voisins_X[x1, :]) * reg_absA[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        )
    )

 # - regularization
    @variable(
        modelA,
        reg_absB[
            x1 in 1:nbX,
            x2 in findall(voisins_X[x1, :]),
            y in Ylevels,
            z in Zlevels,
        ] >= 0
    )
    @constraint(
        modelA,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), x in in 1:nbX, y in Ylevels, z in Zlevels],
        reg_absB[x1, x2, y, z] >=
        gammaA[x, y, x1,z] / (max(1, length(indXB[x1])) / nB) -
        gammaA[x, y, x2,z] / (max(1, length(indXB[x2])) / nB)
    )
    @constraint(
        modelA,
        [x1 in 1:nbX, x2 in findall(voisins_X[x1, :]), x in in 1:nbX, y in Ylevels, z in Zlevels],
        reg_absB[x1, x2, y, z] >=
        gammaA[x, y,x2, z] / (max(1, length(indXB[x2])) / nB) -
        gammaA[x, y, x1,z] / (max(1, length(indXB[x1])) / nB)
    )
    @expression(
        modelA,
        regterm,
        sum(
            1 / length(voisins_X[x1, :]) * reg_absB[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        )
    )

    
    # by default, the OT cost and regularization term are weighted to lie in the same interval
    @objective(
        modelA,
        Min,
        sum(C[x1,y, x2,z] * gammaA[x1, y,x2, z] for y in Ylevels, z in Zlevels, x1 = 1:nbX, x2 = 1:nbX) +
        lambda_reg * sum(
            1 / length(voisins_X[x1, :]) * reg_absA[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        ) +
        lambda_reg * sum(
            1 / length(voisins_X[x1, :]) * reg_absB[x1, x2, y, z] for x1 = 1:nbX,
            x2 in findall(voisins_X[x1, :]), y in Ylevels, z in Zlevels
        )
    )

##Attention au cout pas la meme forme

 

    # Solve the problem
    optimize!(modelA)

    # Extract the values of the solution
    gammaA_val = [value(gammaA[x1, y,x2, z]) for x1 = 1:nbX, y in Ylevels,x2 = 1:nbX, z in Zlevels]

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Julia 1.11.2
#     language: julia
#     name: julia-1.11
# ---

using OptimalTransportDataIntegration
import CSV
using DataFrames
using JuMP, Clp
import .Iterators: product
import Distances: colwise, pairwise, Hamming, Cityblock
import PythonOT

onecold(X) = map(argmax, eachrow(X))

function loss_crossentropy(Y, F)
    ϵ = 1e-12
    res = zeros(size(Y, 1), size(F, 1))
    logF = log.(F .+ ϵ)
    for i in axes(Y, 2)
        res .+= -Y[:, i] .* logF[:, i]'
    end
    return res
end

"""
    optimal_modality(values, loss, weight)

- values: vector of possible values
- weight: vector of weights 
- loss: matrix of size len(Weight) * len(Values)

- Returns an argmin over value in values of the scalar product <loss[value,],weight> 
"""
function optimal_modality(values, loss, weight)

    @assert size(loss, 1) == size(weight, 1)
    @assert size(loss, 2) == size(values, 1)

    cost_for_each_modality = Float64[]
    for j in eachindex(values)
        s = 0
        for i in axes(loss, 1)
            s += loss[i, j] * weight[i]
        end
        push!(cost_for_each_modality, s)
    end

    return values[argmin(cost_for_each_modality)]

end


# -

function unbalanced_solver(data; lambda = 0.0, alpha = 0.0)

    T = Int

    Ylevels = 1:4
    Zlevels = 1:3

    X = Matrix(data[!, [:X1, :X2, :X3]])
    Y = Vector{T}(data.Y)
    Z = Vector{T}(data.Z)

    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    YA = view(Y, indA)
    YB = view(Y, indB)
    ZA = view(Z, indA)
    ZB = view(Z, indB)

    YA_hot = one_hot_encoder(YA, Ylevels)
    ZA_hot = one_hot_encoder(ZA, Zlevels)
    YB_hot = one_hot_encoder(YB, Ylevels)
    ZB_hot = one_hot_encoder(ZB, Zlevels)


    # Compute data for aggregation of the individuals

    nA = length(indA)
    nB = length(indB)

    X_hot = one_hot_encoder(X)
    Xlevels_hot = sort(unique(eachrow(X_hot))) 

    @assert length(Xlevels_hot) == 24 "not possible observations in X"

    Ylevels_hot = one_hot_encoder(Ylevels)
    Zlevels_hot = one_hot_encoder(Zlevels)

    nx = size(X_hot, 2)
    XYAlevels = Vector{Int}[]
    XZBlevels = Vector{Int}[]
    for (y, x) in product(Ylevels, Xlevels_hot)
        push!(XYAlevels, [x...; y])
    end
    for (z, x) in product(Zlevels, Xlevels_hot)
        push!(XZBlevels, [x...; z])
    end

    a = stack([v[1:nx] for v in XYAlevels], dims = 1)
    b = stack([v[1:nx] for v in XZBlevels], dims = 1)

    C0 = pairwise(Hamming(), a, b; dims = 1)
    C = zeros(size(C0))

    c = zeros(length(Xlevels_hot), length(Ylevels), length(Xlevels_hot), length(Zlevels))

    steps = 10

    for step in 1:steps

        for (i, (y, x1)) in enumerate(product(Ylevels, eachindex(Xlevels_hot)))
            for (j, (z, x2)) in enumerate(product(Zlevels, eachindex(Xlevels_hot)))
                c[x1, y, x2, z] = C[i, j]
            end
        end

        Xvalues = unique(eachrow(X))
        dist_X = pairwise(Cityblock(), Xvalues, Xvalues)
        voisins_X = findall.(eachrow(dist_X .<= 1))
        nX = length(Xvalues)


        # Compute the indexes of individuals with same covariates
        indXA = Vector{T}[]
        indXB = Vector{T}[]
        # -

        a = view(X_hot, indA, :)
        b = view(X_hot, indB, :)

        for (i, x) in enumerate(Xlevels_hot)
            distA = vec(pairwise(Hamming(), x[:, :], a', dims = 2))
            distB = vec(pairwise(Hamming(), x[:, :], b', dims = 2))
            push!(indXA, findall(distA .< 0.1))
            push!(indXB, findall(distB .< 0.1))
        end


        # Compute the estimators that appear in the model

        estim_XA = length.(indXA) ./ nA
        estim_XB = length.(indXB) ./ nB
        estim_XA_YA =
            [length(indXA[x][YA[indXA[x]].==y]) / nA for x in eachindex(indXA), y in Ylevels]
        estim_XB_ZB =
            [length(indXB[x][ZB[indXB[x]].==z]) / nB for x in eachindex(indXB), z in Zlevels]


        Xlevels = eachindex(Xlevels_hot)

        model = Model(Clp.Optimizer)
        set_optimizer_attribute(model, "LogLevel", 0)


        @variable(
            model,
            Ω[x1 in Xlevels, y in Ylevels, x2 in Xlevels, z in Zlevels] >= 0,
            base_name = "Ω"
        )
        @variable(model, error_XY[x1 in Xlevels, y in Ylevels], base_name = "error_XY")
        @variable(
            model,
            abserror_XY[x1 in Xlevels, y in Ylevels] >= 0,
            base_name = "abserror_XY"
        )
        @variable(model, error_XZ[x2 in Xlevels, z in Zlevels], base_name = "error_XZ")
        @variable(
            model,
            abserror_XZ[x2 in Xlevels, z in Zlevels] >= 0,
            base_name = "abserror_XZ"
        )

        # - assign sufficient probability to each class of covariates with the same outcome
        @constraint(
            model,
            ctYandXinA[x1 in Xlevels, y in Ylevels],
            sum(Ω[x1, y, x2, z] for x2 in Xlevels, z in Zlevels) ==
            estim_XA_YA[x1, y] + error_XY[x1, y]
        )

        # - we impose that the probability of Y conditional to X is the same in the two databases
        # - the consequence is that the probability of Y and Z conditional to Y is also the same in the two bases
        @constraint(
            model,
            ctZandXinA[x2 in Xlevels, z in Zlevels],
            sum(Ω[x1, y, x2, z] for x1 in Xlevels, y in Ylevels) ==
            estim_XB_ZB[x2, z] + error_XZ[x2, z]
        )

        # - recover the norm 1 of the error
        @constraint(model, [x1 in Xlevels, y in Ylevels], error_XY[x1, y] <= abserror_XY[x1, y])
        @constraint(
            model,
            [x1 in Xlevels, y in Ylevels],
            -error_XY[x1, y] <= abserror_XY[x1, y]
        )
        @constraint(
            model,
            sum(abserror_XY[x1, y] for x1 in Xlevels, y in Ylevels) <= alpha / 2.0
        )
        @constraint(model, sum(error_XY[x1, y] for x1 in Xlevels, y in Ylevels) == 0.0)
        @constraint(model, [x2 in Xlevels, z in Zlevels], error_XZ[x2, z] <= abserror_XZ[x2, z])
        @constraint(
            model,
            [x2 in Xlevels, z in Zlevels],
            -error_XZ[x2, z] <= abserror_XZ[x2, z]
        )
        @constraint(
            model,
            sum(abserror_XZ[x2, z] for x2 in Xlevels, z in Zlevels) <= alpha / 2.0
        )
        @constraint(model, sum(error_XZ[x2, z] for x2 in Xlevels, z in Zlevels) == 0.0)

        # - regularization
        @variable(
            model,
            reg_absA[x1 in Xlevels, x2 in voisins_X[x1], y in Ylevels, z in Zlevels] >= 0
        )
        @constraint(
            model,
            [x1 in Xlevels, x2 in voisins_X[x1], y in Ylevels, x in Xlevels, z in Zlevels],
            reg_absA[x1, x2, y, z] >=
            Ω[x1, y, x, z] / (max(1, length(indXA[x1])) / nA) -
            Ω[x2, y, x, z] / (max(1, length(indXA[x2])) / nA)
        )
        @constraint(
            model,
            [x1 in Xlevels, x2 in voisins_X[x1], y in Ylevels, x in Xlevels, z in Zlevels],
            reg_absA[x1, x2, y, z] >=
            Ω[x2, y, x, z] / (max(1, length(indXA[x2])) / nA) -
            Ω[x1, y, x, z] / (max(1, length(indXA[x1])) / nA)
        )
        @expression(
            model,
            regtermA,
            sum(
                1 / nX * reg_absA[x1, x2, y, z] for x1 in Xlevels, x2 in voisins_X[x1],
                y in Ylevels, z in Zlevels
            )
        )

        # - regularization
        @variable(
            model,
            reg_absB[x1 in Xlevels, x2 in voisins_X[x1], y in Ylevels, z in Zlevels] >= 0
        )
        @constraint(
            model,
            [x1 in Xlevels, x2 in voisins_X[x1], x in Xlevels, y in Ylevels, z in Zlevels],
            reg_absB[x1, x2, y, z] >=
            Ω[x, y, x1, z] / (max(1, length(indXB[x1])) / nB) -
            Ω[x, y, x2, z] / (max(1, length(indXB[x2])) / nB)
        )
        @constraint(
            model,
            [x1 in Xlevels, x2 in voisins_X[x1], x in Xlevels, y in Ylevels, z in Zlevels],
            reg_absB[x1, x2, y, z] >=
            Ω[x, y, x2, z] / (max(1, length(indXB[x2])) / nB) -
            Ω[x, y, x1, z] / (max(1, length(indXB[x1])) / nB)
        )
        @expression(
            model,
            regtermB,
            sum(
                1 / nX * reg_absB[x1, x2, y, z] for x1 in Xlevels, x2 in voisins_X[x1],
                y in Ylevels, z in Zlevels
            )
        )

        # by default, the OT cost and regularization term are weighted to lie in the same interval
        @objective(
            model,
            Min,
            sum(c[x1, y, x2, z] * Ω[x1, y, x2, z] for y in Ylevels, z in Zlevels, x1 = Xlevels, x2 = Xlevels) +
            lambda * sum(
                1 / nX * reg_absA[x1, x2, y, z] for x1 = Xlevels,
                x2 in voisins_X[x1], y in Ylevels, z in Zlevels
            ) +
            lambda * sum(
                1 / nX * reg_absB[x1, x2, y, z] for x1 = Xlevels,
                x2 in voisins_X[x1], y in Ylevels, z in Zlevels
            )
        )


        # Solve the problem
        optimize!(model)


        # Extract the values of the solution
        gamma_val = [
            value(Ω[x1, y, x2, z]) for x1 in Xlevels, y in Ylevels, x2 in Xlevels, z in Zlevels
        ]


        # compute G from gamma_val ?

        G = zeros(size(C0))

        for p1 in axes(G, 1)
            for p2 in axes(G, 2)
                x1 = findfirst(==(XYAlevels[p1][1:nx]), Xlevels_hot)
                y = last(XYAlevels[p1])
                x2 = findfirst(==(XZBlevels[p2][1:nx]), Xlevels_hot)
                z = last(XZBlevels[p2])
                G[p1, p2] = gamma_val[x1, y, x2, z]
            end
        end

        yA = last.(XYAlevels) # les y parmi les XYA observés, potentiellement des valeurs repetées 
        yA_hot = one_hot_encoder(yA, Ylevels)
        zB = last.(XZBlevels) # les z parmi les XZB observés, potentiellement des valeurs repetées 
        zB_hot = one_hot_encoder(zB, Zlevels)

        Yloss = loss_crossentropy(yA_hot, Ylevels_hot)
        Zloss = loss_crossentropy(zB_hot, Zlevels_hot)

        yB_pred = [ optimal_modality(Ylevels, Yloss, view(G, :, j)) for j in axes(G, 2)]
        zA_pred = [ optimal_modality(Zlevels, Zloss, view(G, i, :)) for i in axes(G, 1)]


        yB_pred_hot = one_hot_encoder(yB_pred, Ylevels)
        zA_pred_hot = one_hot_encoder(zA_pred, Zlevels)

        zA_pred_hot_i = zeros(T, (nA, length(Zlevels)))
        yB_pred_hot_i = zeros(T, (nB, length(Ylevels)))

        XA_hot = view(X_hot, indA, :)
        XB_hot = view(X_hot, indB, :)

        XYA = hcat(XA_hot, YA)
        XZB = hcat(XB_hot, ZB)

        for i in 1:nA
           ind = findfirst(XYA[i, :] ≈ v for v in XYAlevels)
           zA_pred_hot_i[i, :] .= zA_pred_hot[ind, :]
        end

        for i in 1:nB
           ind = findfirst(XZB[i, :] ≈ v for v in XZBlevels)
           yB_pred_hot_i[i, :] .= yB_pred_hot[ind, :]
        end

        ### Update Cost matrix
        alpha1 = 1 / maximum(loss_crossentropy(Ylevels_hot, Ylevels_hot))
        alpha2 = 1 / maximum(loss_crossentropy(Zlevels_hot, Zlevels_hot))

        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
        fcost = chinge1 .+ chinge2'

        @show sum(fcost .* G)

        C .= C0 ./ maximum(C0) .+ fcost

        YBpred = onecold(yB_pred_hot_i)
        ZApred = onecold(zA_pred_hot_i)

        est = (sum(YB .== YBpred) .+ sum(ZA .== ZApred)) ./ (nA + nB)

        println("est = $est ")

    end

end

#csv_file = joinpath("dataset.csv")
#data = CSV.read(csv_file, DataFrame)

params = DataParameters(nA = 1000, nB = 1000, mB = [2, 0, 0], p = 0.2)

rng = DataGenerator(params)
data = generate(rng)

@time println(unbalanced_solver(data, lambda = 0.01, alpha = 0.1 ))
#@time println(otrecod(data, JointOTWithinBase(lambda = 0.1, alpha = 0.1)))
@time println(otrecod(data, JointOTBetweenBases()))

params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], p = 0.4)
rng = DataGenerator(params)
data = generate(rng)

@time println(unbalanced_solver(data, lambda = 0.01, alpha = 0.1 ))
# @time println(otrecod(data, JointOTWithinBase(lambda = 0.1, alpha = 0.1)))

@time println(otrecod(data, JointOTBetweenBases()))

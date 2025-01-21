using CSV
using DataFrames
import PythonOT
import .Iterators: product
import Distances: pairwise, Hamming

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

export unbalanced_modality

function unbalanced_modality(
    data,
    reg,
    reg_m;
    Ylevels = 1:4,
    Zlevels = 1:3,
    iterations = 1
)

    T = Int32

    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    X = Matrix{T}(one_hot_encoder(data[!, [:X1, :X2, :X3]]))
    Y = Vector{T}(data.Y)
    Z = Vector{T}(data.Z)

    YA = view(Y, indA)
    YB = view(Y, indB)
    ZA = view(Z, indA)
    ZB = view(Z, indB)

    XA = view(X, indA, :)
    XB = view(X, indB, :)

    YA_hot = one_hot_encoder(YA, Ylevels)
    ZA_hot = one_hot_encoder(ZA, Zlevels)
    YB_hot = one_hot_encoder(YB, Ylevels)
    ZB_hot = one_hot_encoder(ZB, Zlevels)

    XYA = hcat(XA, YA)
    XZB = hcat(XB, ZB)

    distance = Hamming()

    # Compute data for aggregation of the individuals

    nA = length(indA)
    nB = length(indB)

    # list the distinct modalities in A and B
    indY = Dict((m, findall(YA .== m)) for m in Ylevels)
    indZ = Dict((m, findall(ZB .== m)) for m in Zlevels)

    # Compute the indexes of individuals with same covariates
    indXA = Dict{T,Array{T}}()
    indXB = Dict{T,Array{T}}()
    Xlevels = sort(unique(eachrow(X)))

    for (i, x) in enumerate(Xlevels)
        distA = vec(pairwise(distance, x[:, :], XA', dims = 2))
        distB = vec(pairwise(distance, x[:, :], XB', dims = 2))
        indXA[i] = findall(distA .< 0.1)
        indXB[i] = findall(distB .< 0.1)
    end

    nbX = length(indXA)

    wa = vec([sum(indXA[x][YA[indXA[x]].==y]) for y in Ylevels, x = 1:nbX])
    wb = vec([sum(indXB[x][ZB[indXB[x]].==z]) for z in Zlevels, x = 1:nbX])

    wa2 = filter(>(0), wa) ./ nA
    wb2 = filter(>(0), wb) ./ nB

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

    nx = size(X, 2) ## Nb modalités x 

    # les x parmi les XYA observés, potentiellement des valeurs repetées 
    XA_hot = stack([v[1:nx] for v in XYA2], dims = 1)
    # les x dans XZB observés, potentiellement des valeurs repetées 
    XB_hot = stack([v[1:nx] for v in XZB2], dims = 1)

    yA = last.(XYA2) # les y parmi les XYA observés, potentiellement des valeurs repetées 
    yA_hot = one_hot_encoder(yA, Ylevels)
    zB = last.(XZB2) # les z parmi les XZB observés, potentiellement des valeurs repetées 
    zB_hot = one_hot_encoder(zB, Zlevels)

    # Algorithm

    ## Initialisation 

    yB_pred = zeros(size(XZB2, 1)) # number of observed different values in A
    zA_pred = zeros(size(XYA2, 1)) # number of observed different values in B
    nbrvarX = size(data, 2) - 3 # size of data less Y, Z and database id

    dimXZB = length(XZB2[1])
    dimXYA = length(XYA2[1])

    Yloss = loss_crossentropy(yA_hot, Ylevels_hot)
    Zloss = loss_crossentropy(zB_hot, Zlevels_hot)

    alpha1 = 1 / maximum(loss_crossentropy(Ylevels_hot, Ylevels_hot))
    alpha2 = 1 / maximum(loss_crossentropy(Zlevels_hot, Zlevels_hot))

    ## Optimal Transport

    C0 = pairwise(Hamming(), XA_hot, XB_hot; dims = 1) .* nx ./ nbrvarX
    C = C0 ./ maximum(C0)

    zA_pred_hot_i = zeros(T, (nA, length(Zlevels)))
    yB_pred_hot_i = zeros(T, (nB, length(Ylevels)))

    est_opt = 0.0

    YBpred = zeros(T, nB)
    ZApred = zeros(T, nA)
    sav_totalcost = []
    sav_fcost = []
    for iter = 1:iterations

        if reg_m > 0.0
            G = PythonOT.mm_unbalanced(wa2, wb2, C, reg_m; reg = reg, div = "kl")
        else
            G = PythonOT.emd(wa2, wb2, C)
        end

        for j in eachindex(yB_pred)
            yB_pred[j] = optimal_modality(Ylevels, Yloss, view(G, :, j))
        end

        for i in eachindex(zA_pred)
            zA_pred[i] = optimal_modality(Zlevels, Zloss, view(G, i, :))
        end

        yB_pred_hot = one_hot_encoder(yB_pred, Ylevels)
        zA_pred_hot = one_hot_encoder(zA_pred, Zlevels)

        ### Update Cost matrix

        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
        fcost = chinge1 .+ chinge2'

        C .= C0 ./ maximum(C0) .+ fcost

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

        est = (sum(YB .== YBpred) .+ sum(ZA .== ZApred)) ./ (nA + nB)

        est_opt = max(est_opt, est)
        sav_totalcost.append(np.sum(G*C))
        sav_fcost.append(np.sum(G*fcost))
    end

    return est_opt


end

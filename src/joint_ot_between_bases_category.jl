using CSV
using DataFrames
import PythonOT
import .Iterators: product
import LinearAlgebra: norm

function joint_ot_between_bases_category(
        data,
        reg,
        reg_m1,
        reg_m2;
        Ylevels = 1:4,
        Zlevels = 1:3,
        iterations = 1
    )

    T = Int32

    function loss_crossentropy(Y::AbstractMatrix{T}, F::AbstractMatrix{T}) where {T}
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
    
    function modality_cost(loss, weight)
    
        cost_for_each_modality = Float64[]
        for j in axes(loss, 2)
            s = zero(Float64)
            for i in axes(loss, 1)
                s += loss[i, j] * weight[i]
            end
            push!(cost_for_each_modality, s)
        end
    
        return Flux.softmax(cost_for_each_modality)
    
    end 


    base = data.database

    indA = findall(base .== 1)
    indB = findall(base .== 2)

    colnames = names(data, r"^X")
    
    X = Matrix{Float32}(data[!, colnames])
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

    # Compute data for aggregation of the individuals

    nA = length(indA)
    nB = length(indB)

    # list the distinct modalities in A and B
    indY = Dict((m, findall(YA .== m)) for m in Ylevels)
    indZ = Dict((m, findall(ZB .== m)) for m in Zlevels)

    # Compute the indexes of individuals with same covariates
    indXA = Dict{T, Array{T}}()
    indXB = Dict{T, Array{T}}()
    Xlevels = sort(unique(eachrow(X)))

    distance = Euclidean()

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

    Ylevels_hot = one_hot_encoder(Ylevels)
    Zlevels_hot = one_hot_encoder(Zlevels)

    nx = size(X, 2) ## Nb modalités x

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

    Yloss = loss_crossentropy(yA_hot, Ylevels_hot)
    Zloss = loss_crossentropy(zB_hot, Zlevels_hot)

    alpha1 = 1 / maximum(loss_crossentropy(Ylevels_hot, Ylevels_hot))
    alpha2 = 1 / maximum(loss_crossentropy(Zlevels_hot, Zlevels_hot))

    ## Optimal Transport
    C0 = Float32.(pairwise(SqEuclidean(), XA_hot, XB_hot, dims = 1))
    C0 .= C0 ./ maximum(C0)
    C = copy(C0)
  
    zA_pred_hot_i = zeros(T, (nA, length(Zlevels)))
    yB_pred_hot_i = zeros(T, (nB, length(Ylevels)))

    est_opt = 0.0

    YBpred = zeros(T, nB)
    ZApred = zeros(T, nA)

    G = ones(Float32, length(wa2), length(wb2))
    Gold = copy(G)
    cost = Inf

    for iter in 1:iterations

        costold = cost

        if reg_m1 > 0.0 && reg_m2 > 0.0
            G .= PythonOT.mm_unbalanced(wa2, wb2, C, (reg_m1, reg_m2); reg = reg, div = "kl")
        else
            G .= PythonOT.emd(wa2, wb2, C)
        end

        delta = norm(G .- Gold)

        Gold .= G

        for j in eachindex(yB_pred)
            yB_pred[j] = Ylevels[argmin(modality_cost(Yloss, view(G, :, j)))]
        end

        for i in eachindex(zA_pred)
            zA_pred[i] = Zlevels[argmin(modality_cost(Zloss, view(G, i, :)))]
        end

        yB_pred_hot = one_hot_encoder(yB_pred, Ylevels)
        zA_pred_hot = one_hot_encoder(zA_pred, Zlevels)

        ### Update Cost matrix

        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
    
        fcost = chinge1.^2 .+ chinge2'.^2

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

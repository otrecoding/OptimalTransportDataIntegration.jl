using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf
using CSV
using DataFrames
using PythonOT
using .Iterators: product
using Distances: pairwise, Hamming, Euclidean
using LinearAlgebra: norm

onecold(X) = map(argmax, eachrow(X))

"""
    loss_crossentropy(Y, F)

Cross entropy is typically used as a loss in multi-class classification, in which case the labels y are given in a one-hot format. dims specifies the dimension (or the dimensions) containing the class probabilities. The prediction ŷ is usually probabilities but in our case it is also one hot encoded vector.

"""
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


"""
    modality_cost(loss, weight)

- loss: matrix of size len(weight) * len(levels)
- weight: vector of weights 

Returns the scalar product <loss[level,],weight> 
"""
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
params = DataParameters(
    nA = 1000,
    nB = 1000,
    mA = [0.0],
    mB = [2.0],
    covA = ones(1,1),
    covB = ones(1,1),
    aA = [1.0],
    aB = [1.0],
    r2 = 0.9,
    pA = [[0.5, 0.5]],
    pB= [[ 0.8, 0.2]]  )
 
rng = DiscreteDataGenerator(params; scenario = 2)
data = generate(rng)

reg=0.0
reg_m1=0
reg_m2=0
Ylevels = 1:4
Zlevels = 1:3
iterations = 1
T = Int32

base = data.database
indA = findall(base .== 1)
indB = findall(base .== 2)

colnames = names(data, r"^X")
    #X_hot = Matrix{T}(one_hot_encoder(data[!, colnames]))
X_hot = Matrix{T}(data[!, colnames])
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
        distA = vec(pairwise(Euclidean(), x[:, :], XA', dims = 2))
        distB = vec(pairwise(Euclidean(), x[:, :], XB', dims = 2))
        indXA[i] = findall(distA .< 0.1)
        indXB[i] = findall(distB .< 0.1)
    end

nbXA = length(indXA)
nbXB = length(indXB)

wa = vec([length(indXA[x][findall(YA[indXA[x]] .== y)]) / nA for y in Ylevels, x in 1:nbXA])
wb = vec([length(indXB[x][findall(ZB[indXB[x]] .== z)]) / nB for z in Zlevels, x in 1:nbXB])

wa2 = filter(>(0), wa)
wb2 = filter(>(0), wb) 

XZB2 = Vector{T}[]
XYA2 = Vector{T}[]
j = 0

for (y, x) in product(Ylevels, Xlevels)
        global j
        j += 1
        wa[j] > 0 && push!(XYA2, [x...; y])
end
k = 0
for (z, x) in product(Zlevels, Xlevels)
    global k
    k += 1
    wb[k] > 0 && push!(XZB2, [x...; z])
end

Ylevels_hot = one_hot_encoder(Ylevels)
Zlevels_hot = one_hot_encoder(Zlevels)

nx = size(X_hot, 2) ## Nb modalités x
display(nx)
    # les x parmi les XYA observés, potentiellement des valeurs repetées
XA_hot = stack([v[1:nx] for v in XYA2], dims = 1)
    # les x parmi les XZB observés, potentiellement des valeurs repetées
XB_hot = stack([v[1:nx] for v in XZB2], dims = 1)

yA = last.(XYA2) # les y parmi les XYA observés, potentiellement des valeurs repetées
yA_hot = one_hot_encoder(yA, Ylevels)
zB = last.(XZB2) # les z parmi les XZB observés, potentiellement des valeurs repetées
zB_hot = one_hot_encoder(zB, Zlevels)

yB_pred = zeros(T, size(XZB2, 1)) # number of observed different values in A
zA_pred = zeros(T, size(XYA2, 1)) # number of observed different values in B

dimXZB = length(XZB2[1])
dimXYA = length(XYA2[1])

Yloss = loss_crossentropy(yA_hot, Ylevels_hot)
Zloss = loss_crossentropy(zB_hot, Zlevels_hot)

alpha1 = 1 / maximum(loss_crossentropy(Ylevels_hot, Ylevels_hot))
alpha2 = 1 / maximum(loss_crossentropy(Zlevels_hot, Zlevels_hot))

    ## Optimal Transport
#XA_hot.= Float64.(XA_hot)
#XB_hot.= Float64.(XB_hot)
C0 = pairwise(Euclidean(), XA_hot, XB_hot, dims = 1)
    
C0 = C0 ./ maximum(C0)
C0 .= C0.^2
C = C0
  
zA_pred_hot_i = zeros(T, (nA, length(Zlevels)))
yB_pred_hot_i = zeros(T, (nB, length(Ylevels)))

est_opt = 0.0

YBpred = zeros(T, nB)
ZApred = zeros(T, nA)

G = ones(length(wa2), length(wb2))
display(G)
cost = Inf
display(C)
display(wa2)
display(wb2)


G = PythonOT.emd(wa2, wb2,  C)


    
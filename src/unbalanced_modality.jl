using OptimalTransportDataIntegration
using OTRecod
import PythonOT
import .Iterators: product
import Distances: pairwise, Hamming

onecold(X) = map(argmax, eachrow(X))
    
function loss_crossentropy(Y, F)
    ϵ = 1e-12
    res = zeros(size(Y,1), size(F,1))
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
    
    cost_for_each_modality=Float64[]
    for j in eachindex(values)
        s=0
        for i in axes(loss, 1)
            s += loss[i,j] * weight[i]
        end
        push!(cost_for_each_modality, s)
    end
        
    return values[argmin(cost_for_each_modality)]

end

function unbalanced_modality( data; iterations = 1)
    
    database = data.database
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))

    nA = size(dba, 1)
    nB = size(dbb, 1)

    YB_true = dbb.Y
    ZA_true = dba.Z
    
    Xnames = ["X1", "X2", "X3"]
    XA = one_hot_encoder(Matrix(dba[!, Xnames]))
    XB = one_hot_encoder(Matrix(dbb[!, Xnames]))

    XYA_i = hcat(XA, dba.Y)
    XZB_i = hcat(XB, dbb.Z)

    X = vcat(XA, XB)
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    instance = OTRecod.Instance( database, X, Y, Z, Hamming())

    lambda_reg = 0.392
    maxrelax = 0.714
    percent_closest = 0.2
    
    sol = ot_joint(instance, maxrelax, lambda_reg, percent_closest)
    OTRecod.compute_pred_error!(sol, instance, false)

    
    # Compute data for aggregation of the individuals

    indXA = instance.indXA
    indXB = instance.indXB
    nbX = length(indXA)

    wa = vec([length(indXA[x][findall(instance.Yobserv[indXA[x]] .== y)]) / nA for y in instance.Y, x = 1:nbX])
    wb = vec([length(indXB[x][findall(instance.Zobserv[indXB[x] .+ nA] .== z)]) / nB for z in instance.Z, x = 1:nbX ])  
    wa2 = wa[wa .> 0.0]
    wb2 = wb[wb .> 0.0]

    Xvalues = stack(sort(unique(eachrow(one_hot_encoder(instance.Xval)))), dims=1)

    Xobserv = sort(unique(eachrow(instance.Xobserv)))
    Yobserv = sort(unique(instance.Yobserv))
    Zobserv = sort(unique(instance.Zobserv))

    XYA = Vector{Int}[]
    for (y,x) in product(Yobserv,Xobserv)
        push!(XYA, [x...; y])
    end
    XZB = Vector{Int}[]
    for (z,x) in product(Zobserv,Xobserv)
        push!(XZB, [x...; z])
    end

    XYA2 = XYA[wa .> 0] # XYA observés
    XZB2 = XZB[wb .> 0] # XZB observés

    Y_hot = one_hot_encoder(instance.Y)
    Z_hot = one_hot_encoder(instance.Z)

    nx = size(Xvalues, 2) # Nb modalités x 

    XA_hot = stack([v[1:nx] for v in XYA2], dims=1) # les x parmi les XYA observés, potentiellement des valeurs repetées 
    XB_hot = stack([v[1:nx] for v in XZB2], dims=1) # les x parmi les XZB observés, potentiellement des valeurs repetées 

    yA = getindex.(XYA2, nx+1) # les y parmi les XYA observés, potentiellement des valeurs repetées 
    yA_hot = one_hot_encoder(yA)
    zB = getindex.(XZB2, nx+1) # les z parmi les XZB observés, potentiellement des valeurs repetées 
    zB_hot = one_hot_encoder(zB)

    nbrvarX = 3

    dimXZB = length(XZB2[1])
    dimXYA = length(XYA2[1])

    yB_pred = zeros(size(XZB2, 1)) # number of observed different values in B
    zA_pred = zeros(size(XYA2, 1)) # number of observed different values in A
    
    Y_loss = loss_crossentropy(yA_hot, Y_hot)
    Z_loss = loss_crossentropy(zB_hot, Z_hot) 

    ### Optimal Transport

    C0 = pairwise(Hamming(), XA_hot, XB_hot; dims=1) .* nx ./ nbrvarX
    C = C0 ./ maximum(C0)

    zA_pred_hot_i = zeros(Int, (nA,length(instance.Z)))
    yB_pred_hot_i = zeros(Int, (nB,length(instance.Y)))

    est = 0.0

    for iter in 1:iterations
    
        G = PythonOT.mm_unbalanced(wa2, wb2, C, 0.1; div="kl") #unbalanced
    
        for j in eachindex(yB_pred)
            yB_pred[j] = optimal_modality(instance.Y, Y_loss, G[:,j])
        end
        for i in eachindex(zA_pred)
            zA_pred[i] = optimal_modality(instance.Z, Z_loss, G[i,:])
        end
    
        yB_pred_hot = one_hot_encoder(yB_pred)
        zA_pred_hot = one_hot_encoder(zA_pred)
 
        ### Update Cost matrix
        alpha1 = 1 / maximum(loss_crossentropy(yA_hot, yB_pred_hot))
        alpha2 = 1 / maximum(loss_crossentropy(zB_hot, zA_pred_hot))
 
        chinge1 = alpha1 * loss_crossentropy(yA_hot, yB_pred_hot)
        chinge2 = alpha2 * loss_crossentropy(zB_hot, zA_pred_hot)
        fcost = chinge1 .+ chinge2'
 
        C .= C0 ./ maximum(C0) .+ fcost
 
        ### Predict
    
        for i in axes(XYA_i, 1)
            ind = findfirst(XYA_i[i,:] == v for v in XYA2)
            zA_pred_hot_i[i,:] .= zA_pred_hot[ind,:]
        end

        for i in axes(XZB_i, 1)
            ind = findfirst(XZB_i[i,:] == v for v in XZB2)
            yB_pred_hot_i[i,:] .= yB_pred_hot[ind,:]
        end
 
        YB_pred = onecold(yB_pred_hot_i) 
        ZA_pred = onecold(zA_pred_hot_i)
 
        ### Evaluate 
 
        est = (sum(YB_true .== YB_pred) .+ sum(ZA_true .== ZA_pred)) ./ (nA + nB)

    end

    return est

end
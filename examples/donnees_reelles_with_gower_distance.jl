import Flux
import Flux: Chain, Dense
using FreqTables
using LinearAlgebra
import PythonOT

include(joinpath(@__DIR__, "read_data.jl"))
include(joinpath(@__DIR__, "gower_distance.jl"))

function main_between_with_gower_distance()

    data = read_data()
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    
    colnames = names(data, r"^X")
    XA = transpose(Matrix{Float32}(dba[!, colnames]))
    XB = transpose(Matrix{Float32}(dbb[!, colnames]))
    
    Ylevels = 1:4
    Zlevels = 1:6
    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)
    
    XYA = vcat(XA, YA)
    XZB = vcat(XB, ZB)
    dimXYA = size(XYA, 1)
    dimXZB = size(XZB, 1)
    nA = size(dba, 1)
    nB = size(dbb, 1)
    
    wa = ones(Float32, nA) ./ nA
    wb = ones(Float32, nB) ./ nB
    iterations = 10
    learning_rate = 0.01
    batchsize = 444
    epochs = 1000
    hidden_layer_size = 10
    
    # Transport des covariables - choix du rho optimal
    
    XA1=dba[:,1]
    XA2=dba[:,2].+1
    XA3=dba[:,3].+1
    XA4=dba[:,4]
    XA5=dba[:,5]
    XA2f = Flux.onehotbatch(XA2, 1:2)
    XA3f = Flux.onehotbatch(XA3, 1:2)
    XA5f = Flux.onehotbatch(XA5, 1:8)
    
    reg_m1_list = [0.01, 0.1,0.5, 1.0]
    reg_m2_list = [0.01, 0.1, 0.5,1.0]
    
    results = []   # vecteur vide pour stocker les tuples (reg_m1, reg_m2, cost)
    cols = names(dba, r"^X") 
    A = dba[:, cols]
    B = dbb[:, cols]
    C = vcat(A, B)
    dist = GowerDF2([:X1,:X4], [:X2,:X3,:X5],C)  
     
    C0 = Float32.(pairwise_gower(dist, dba, dbb))
    C = C0.^2 ./ maximum(C0.^2)
    
    XB1=dbb[:,1]
    XB2=dbb[:,2].+1
    XB3=dbb[:,3].+1
    XB4=dbb[:,4]
    XB5=dbb[:,5]
    XB2f = Flux.onehotbatch(XB2, 1:2)
    XB3f = Flux.onehotbatch(XB3, 1:2)
    XB5f = Flux.onehotbatch(XB5, 1:8)
    
    # Choix du paramètre rho optimal
    
    # listes de valeurs
    reg_m1_list = [0.01, 0.1,0.5, 1.0]
    reg_m2_list = [0.01, 0.1, 0.5,1.0]
    reg= 0.001
    results = []   # vecteur vide pour stocker les tuples (reg_m1, reg_m2, cost)
    XBpred1 = Vector{Float64}[]
    XBpred2 = Matrix{Float64}[]
    XBpred3 = Matrix{Float64}[]
    XBpred4 = Vector{Float64}[]
    XBpred5 = Matrix{Float64}[]
    XBpred = Matrix{Float64}[]
    XB = DataFrame( X1 = XB1, X2 = XB2, X3 = XB3, X4 = XB4, X5 = XB5)

    #PN for r1 in reg_m1_list
    #PN     for r2 in reg_m2_list
    #PN         G = PythonOT.mm_unbalanced(wa, wb, C, (r1, r2); reg = reg, div = "kl")
    #PN         row_sumsB = vec(sum(G, dims=2))
    #PN         col_sumsA = vec(sum(G, dims=1))
    #PN         XBpred1 = (1 ./col_sumsA) .* vec(XA1' * G)
    #PN         XBpred2f = (XA2f* G) .* (1 ./col_sumsA)'
    #PN         XBpred2 = Flux.onecold(XBpred2f)
    #PN         XBpred3f = (XA3f* G) .* (1 ./col_sumsA)'
    #PN         XBpred3 = Flux.onecold(XBpred3f)
    #PN         XBpred4 =(1 ./col_sumsA) .* vec(XA4' * G)
    #PN         XBpred5f = (XA5f* G) .* (1 ./col_sumsA)'
    #PN         XBpred5 = Flux.onecold(XBpred5f)
    #PN         XBpred = DataFrame( X1 = XBpred1, X2 = XBpred2, X3 = XBpred3, X4 = XBpred4, X5 = XBpred5)
    #PN         cost=0
    #PN         for i in 1:nB
    #PN             cost += Distances.evaluate(dist, XB[i,:], XBpred[i,:])
    #PN         end
    #PN         push!(results, (reg_m1=r1, reg_m2=r2, cost=cost))
    #PN     end
    #PN end
    #PN best = findmin([r.cost for r in results])  # retourne (valeur_min, index)
    #PN best_value, idx = best
    #PN best_params = results[idx]
    #PN 
    #PN println("Meilleur coût = ", best_value)
    #PN println("Paramètres optimaux = ", best_params)
    
    r1= 0.01
    r2= 0.01
    reg= 0.001
    G = PythonOT.mm_unbalanced(wa, wb, C, (r1, r2); reg = reg, div = "kl")
    row_sumsB = vec(sum(G, dims=2))
    col_sumsA = vec(sum(G, dims=1))
    
    XBpred1 = (1 ./col_sumsA) .* vec(XA1' * G)
    XBpred2f = (XA2f* G) .* (1 ./col_sumsA)'
    XBpred2 = Flux.onecold(XBpred2f)
    XBpred3f = (XA3f* G) .* (1 ./col_sumsA)'
    XBpred3 = Flux.onecold(XBpred3f)
    XBpred4 =(1 ./col_sumsA) .* vec(XA4' * G)
    XBpred5f = (XA5f* G) .* (1 ./col_sumsA)'
    XBpred5 = Flux.onecold(XBpred5f)
    
    XApred1 = (1 ./row_sumsB) .* vec(XB1' * G')
    XApred2f = (XB2f* G') .* (1 ./row_sumsB)'
    XApred2 = Flux.onecold(XApred2f)
    XApred3f = (XB3f* G') .* (1 ./row_sumsB)'
    XApred3 = Flux.onecold(XApred3f)
    XApred4 =(1 ./row_sumsB) .* vec(XB4' * G')
    XApred5f = (XB5f* G') .* (1 ./row_sumsB)'
    XApred5 = Flux.onecold(XApred5f)
    
    # %%
    XAt = hcat(XA1,XA2f',XA3f',XA4,XA5f')
    XBpredt = hcat(XBpred1,XBpred2f',XBpred3f',XBpred4,XBpred5f')
    XA=XAt'
    XBpred=XBpredt'
    XBt = hcat(XB1,XB2f',XB3f',XB4,XB5f')
    XApredt = hcat(XApred1,XApred2f',XApred3f',XApred4,XApred5f')
    XB=XBt'
    XApred=XApredt'
    
    # %%
    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)
    
    modelXYA = Chain(Dense(dimXA, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    modelXZB = Chain(Dense(dimXB, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    
    function train!(model, x, y)
    
            loader = Flux.DataLoader((x, y), batchsize = batchsize, shuffle = true)
            optim = Flux.setup(Flux.Adam(learning_rate), model)
    
            for epoch in 1:epochs
                for (x, y) in loader
                    grads = Flux.gradient(model) do m
                        y_hat = m(x)
                        Flux.logitcrossentropy(y_hat, y)
                    end
                    Flux.update!(optim, model, grads[1])
                end
            end
    
            return
    end
    
    # %%
    train!(modelXYA, XApred, YA)
    train!(modelXZB, XBpred, ZB)
    ZApred = modelXZB(XA)
    YBpred = modelXYA(XB)
    
    ybpred=Flux.onecold(YBpred)
    zapred= Flux.onecold(ZApred)
    
    # %%
    @show tab1 = FreqTables.freqtable(dba.Y,zapred)
    @show tab2 = FreqTables.freqtable(dbb.Z,ybpred)
    
    
    # Avec la distance de gower
    
    function loss_crossentropy(Y, F)
    
            ϵ = 1.0e-12
            res = zeros(Float32, size(Y, 2), size(F, 2))
            logF = zeros(Float32, size(F))
    
            for i in eachindex(F)
                if F[i] ≈ 1.0
                    logF[i] = log(1.0 - ϵ)
                else
                    logF[i] = log(ϵ)
                end
            end
    
            for i in axes(Y, 1)
                res .+= -Y[i, :] .* logF[i, :]'
            end
    
            return res
    
        end
    
    # %%
    dimXA = size(XA, 1)
    dimXB = size(XB, 1)
    dimYA = size(YA, 1)
    dimZB = size(ZB, 1)
    XA = transpose(Matrix{Float32}(dba[!, colnames]))
    XB = transpose(Matrix{Float32}(dbb[!, colnames]))
    XYA = vcat(XA, YA)
    XZB = vcat(XB, ZB)
    dimXYA = size(XYA, 1)
    dimXZB = size(XZB, 1)
    modelXYA = Chain(Dense(dimXYA, hidden_layer_size), Dense(hidden_layer_size, dimZB))
    modelXZB = Chain(Dense(dimXZB, hidden_layer_size), Dense(hidden_layer_size, dimYA))
    YBpred = modelXZB(XZB)
    ZApred = modelXYA(XYA)
    
    G = ones(Float32, nA, nB)
    
    results = []   # vecteur vide pour stocker les tuples (reg_m1, reg_m2, cost)
    cols = names(dba, r"^X") 
    A = dba[:, cols]
    B = dbb[:, cols]
    C = vcat(A, B)
    dist = GowerDF2([:X1,:X4], [:X2,:X3,:X5],C)  
    
    C0 = Float32.(pairwise_gower(dist, A, B))
    
    C = C0.^2 ./ maximum(C0.^2)
    cost = Inf
    reg = 0.001
    reg_m1 = 0.01
    reg_m2 = 0.01
    
    # %%
    alpha1, alpha2 = 1 / length(Ylevels), 1 / length(Zlevels)
    
    G = ones(Float32, nA, nB)
    cost = Inf
    
    YB = nB .* YA * G
    ZA = nA .* ZB * G'
    for iter in 1:iterations # BCD algorithm
    
            Gold = copy(G)
            costold = cost
            G .= PythonOT.emd(wa, wb, C)
            if reg > 0
                G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
            else
                G .= PythonOT.emd(wa, wb, C)
            end
    
            delta = norm(G .- Gold)
            YB .= nB .* YA * G
            ZA .= nA .* ZB * G'
            train!(modelXYA, XYA, ZA)
            train!(modelXZB, XZB, YB)
    
            YBpred .= modelXZB(XZB)
            ZApred .= modelXYA(XYA)
    
            loss_y = alpha1 * loss_crossentropy(YA, YBpred)
            loss_z = alpha2 * loss_crossentropy(ZB, ZApred)
    
            fcost = loss_y .^ 2 .+ loss_z' .^ 2
    
            cost = sum(G .* fcost)
    
    
            C .= C0 .+ fcost
    
    end
    
    ybpred=Flux.onecold(YBpred)
    zapred= Flux.onecold(ZApred)
    
    @show tab1 = FreqTables.freqtable(dba.Y,zapred)
    @show tab2 = FreqTables.freqtable(dbb.Z,ybpred)
    
end

@time main_between_with_gower_distance()

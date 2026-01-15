# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# +
using XLSX
using DataFrames
using OptimalTransportDataIntegration
using DataFrames, CategoricalArrays
using Distributions
using DocStringExtensions
using Parameters
using Printf
using DataFrames
using LinearAlgebra
using Flux
using Distances
using PythonOT
import Statistics: median
using Pkg
using FreqTables
using DataFrames, Statistics, StatsBase, MultivariateStats
using JuMP, Clp
tbl = XLSX.readtable("examples/data.xlsx", 1)   # 1 = première feuille, ou utiliser le nom "Sheet1"
data = DataFrame(tbl)
eltype.(eachcol(data))
for c in names(data)
    data[!, c] = [ 
        x isa Number ? float(x) :                 # si déjà numérique → convertir en Float64
        x isa Missing ? missing :                 # missing → garder
        tryparse(Float64, String(x))              # si string → tenter de parser
        for x in data[!, c]
    ]
end


data = DataFrame(data)
# -

# Préparation des données

# +
describe(data, :nmissing)



# +
for c in [:Y, :Z]
    data[!, c] = coalesce.(data[!, c], -1)
end

for c in [:Y, :Z, :database]
    data[!, c] = convert.(Union{Int64}, data[!, c])
end
# -

println(data)

# +
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

cols = names(dba, r"^X")   


#Ylevels = 1:5
#Zlevels = 1:6
Ylevels = 1:4
Zlevels = 1:6
YA = Flux.onehotbatch(dba.Y, Ylevels)
ZB = Flux.onehotbatch(dbb.Z, Zlevels)

# -

# Preparation des covariables en plusieurs format. X: discretisé. Xdm discrétisé en gardant la médiane comme valeur X_hot. One hot encoding

# +


digitize(x, bins) = searchsortedlast.(Ref(bins), x)

XA = dba[:, cols]
XB = dbb[:, cols]

X = Vector{Int}[]
Xmdn = Vector{Float64}[]
cols = ["X1", "X4"]
for col in cols

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

cols = ["X2", "X3","X5"]
for col in cols

        X1 = XA[!, col]
        X2 = XB[!, col]
        push!(X, vcat(X1, X2))

        X1mdn = XA[!, col]

        X2mdn = XB[!, col]

        push!(Xmdn, vcat(X1mdn, X2mdn))

end
X = stack(X)
Xmdn = stack(Xmdn)

# +

X_hot = one_hot_encoder(X)

# -

# OT within

# +

Y = Vector(data.Y)
Z = Vector(data.Z)
XA = transpose(Matrix(dba[:, cols]))
XB = transpose(Matrix(dbb[:, cols]))
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

# -

# Distance de Gower

# +
struct GowerDF2 <: PreMetric
    continuous::Vector{Symbol}
    categorical::Vector{Symbol}
    ranges::Dict{Symbol,Float64}   # range par variable continue
end

function GowerDF2(continuous, categorical, df::DataFrame)
    ranges = Dict{Symbol,Float64}()
    for col in continuous
        coldata = skipmissing(df[!, col])
        ranges[col] = maximum(coldata) - minimum(coldata)
     end
    return GowerDF2(continuous, categorical, ranges)
end

function Distances.evaluate(d::GowerDF2, a::DataFrameRow, b::DataFrameRow)
    s = 0.0
    p = length(d.continuous) + length(d.categorical)

    # continues (version correcte Gower)
    for col in d.continuous
        range = d.ranges[col]

        if range == 0 || ismissing(a[col]) || ismissing(b[col])
            contrib = 0.0
        else
            contrib = abs(a[col] - b[col]) / range
        end

        s += contrib
    end

    # catégories
    for col in d.categorical
        if ismissing(a[col]) || ismissing(b[col])
            s += 0.0
        else
            s += (a[col] == b[col] ? 0.0 : 1.0)
        end
    end

    return s / p
end

function pairwise_gower(d::GowerDF2, A::DataFrame, B::DataFrame)
    nA = nrow(A)
    nB = nrow(B)
    D = Matrix{Float64}(undef, nA, nB)

    for i in 1:nA
        row_i = A[i, :]
        for j in 1:nB
            row_j = B[j, :]
            D[i,j] = Distances.evaluate(d, row_i, row_j)
        end
    end

    return D
end

# +

cols = names(dba, r"^X")
for name in ["X2","X3","X5"]
    lev = union(unique(dba[!, name]), unique(dbb[!, name]))
    dba[!, name] = categorical(dba[!, name], levels=lev)
    dbb[!, name] = categorical(dbb[!, name], levels=lev)
end             

# convertir les continues en Float64
for name in ["X1","X4"]
    dba[!, name] = Float64.(dba[!, name])
    dbb[!, name] = Float64.(dbb[!, name])
end
A = dba[:, cols]
B = dbb[:, cols]
C = vcat(A, B)
dist = GowerDF2([:X1,:X4], [:X2,:X3,:X5],C)  
# version pour DataFrameRow
D = pairwise_gower(dist, A, B)
# -

Format coordonnées continues après FADM

# +

# --- Exemple : DataFrames A et B déjà définis ---
df_all = vcat(A, B)

# --- Colonnes continues et catégorielles ---
continuous_cols = [:X1, :X4]
categorical_cols = [:X2, :X3, :X5]

# 1. Standardisation des variables continues
df_cont_std = DataFrame()
for c in continuous_cols
    x = skipmissing(df_all[!, c])
    μ = mean(x)
    σ = std(x)
    df_cont_std[!, c] = (df_all[!, c] .- μ) ./ σ
end

# 2. One-hot encoding des variables catégorielles
df_dummy = DataFrame()
for c in categorical_cols
    df_cat = categorical(df_all[!, c])
    for lvl in levels(df_cat)
        col_name = Symbol(string(c) * "_" * string(lvl))
        df_dummy[!, col_name] = df_cat .== lvl
    end
end

# 3. Pondération FAMD
df_dummy_famd = DataFrame()
for c in names(df_dummy)
    p = mean(df_dummy[!, c])
    df_dummy_famd[!, c] = df_dummy[!, c] ./ sqrt(p)
end

# 4. Construction de la matrice finale
Xn = hcat(Matrix(df_cont_std), Matrix(df_dummy_famd))

# 5. ACP (PCA)
ncomp = 13
model = fit(PCA, Xn'; maxoutdim=ncomp)

# 6. Coordonnées factorielles
coords_all = MultivariateStats.transform(model, Xn')  #
# --- coords_all contient les coordonnées FAMD de df_all ---
coords_all2 = coords_all'
# -

@show size Xn'
@show X

nA = nrow(A)
a = coords_all2[1:nA, :]'
b = coords_all2[nA+1:end, :]'
distance = Euclidean()
D = pairwise(distance, a, b, dims = 2)

# +

lambda = 0.1
alpha = 0.4
percent_closest = 1
#a = Xmdn[indA, :]'
#b = Xmdn[indB, :]'
#a = X_hot[indA, :]'
#b = X_hot[indB, :]'
distance = Euclidean()
#D = pairwise(distance, a, b, dims = 2)


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



# +

tab1 = FreqTables.freqtable(dba.Y,predZA)
tab2 = FreqTables.freqtable(dbb.Z,predYB)
@show round.(tab1 ./ sum(tab1) .* 100, digits = 2)
#@show round.(tab2 ./ sum(tab2) .* 100, digits = 2)
#deuxieme essai avec CSP
#Y codage timoun 1 2 3 4 5
#Y1 primaire ==> Z1
#Y2 secondaire ==> Z2
#Y3 baccalaureat ==> Z3
#Y4 supérieures ==> Z4 Z5 Z6

#accZ=(tab1[1,1]+tab1[2,2]+tab1[3,3]+tab1[4,4]+tab1[4,5])/sum(tab1)
#accZ=(tab1[1,1]+tab1[2,2]+tab1[3,3]+tab1[4,4]+tab1[4,6])/sum(tab1)
#accY=(tab2[1,1]+tab2[2,2]+tab2[3,3]+tab2[4,4]+tab2[5,4]+tab2[6,4])/sum(tab2)
#@show accZ

# +
predZAm = [findmax([probaZindivA[i, z] for z in Zlevels])[1] for i in 1:nA]
predYBm = [findmax([probaYindivB[j, y] for y in Ylevels])[1] for j in 1:nB]

df1 = DataFrame(Y=dba.Y, pred=predZA, prob=predZAm)

# Groupement par combinaison (Y, pred)
stats1 = combine(groupby(df1, [:Y, :pred]),
                 :prob => (x -> round(mean(x), digits=2)) => :mean_prob,
                 :prob => (x -> round(std(x),  digits=2)) => :std_prob)

# +
df1 = DataFrame(Y=dbb.Z, pred=predYB, prob=predYBm)

# Groupement par combinaison (Y, pred)
stats1 = combine(groupby(df1, [:Y, :pred]),
                 :prob => (x -> round(mean(x), digits=2)) => :mean_prob,
                 :prob => (x -> round(std(x),  digits=2)) => :std_prob)

# +
df1 = DataFrame(Y=dba.Y, pred=predZA, prob=predZAm)

# Groupement par combinaison (Y, pred)
stats1 = combine(groupby(df1, [:Y, :pred]),
                 :prob => (x -> round(mean(x), digits=2)) => :mean_prob,
                 :prob => (x -> round(std(x),  digits=2)) => :std_prob)
# -

@show(data)

OT -between

# +
T = Float64
X_hot=Matrix{T}(X_hot)
Xdf = DataFrame(X_hot, Symbol.("X" .* string.(1:size(X_hot, 2))))
data2 = hcat(Xdf,data[:, [:Y, :Z, :database]])
dba = subset(data2, :database => ByRow(==(1)))
dbb = subset(data2, :database => ByRow(==(2)))
@show data2

T = Float64
Xdf = DataFrame(coords_all2, Symbol.("X" .* string.(1:size(coords_all2, 2))))
data3 = hcat(Xdf,data[:, [:Y, :Z, :database]])
dba = subset(data3, :database => ByRow(==(1)))
dbb = subset(data3, :database => ByRow(==(2)))
@show data3

# -

@show data3

@show X

X=Matrix{T}(X)
Xdf = DataFrame(X, Symbol.("X" .* string.(1:size(X, 2))))
data4 = hcat(Xdf,data[:, [:Y, :Z, :database]])


# +
A= otrecod(data2, JointOTBetweenBases(reg = 0.001,
reg_m1= 0.01,
reg_m2 = 0.01,
Ylevels = 1:4,
Zlevels = 1:6))

@show A.yb_pred
@show A.za_pred

# +
A= otrecod(data3, JointOTBetweenBasesWithPredictors(reg = 0.001,
reg_m1= 0.01,
reg_m2 = 0.01,
Ylevels = 1:4,
Zlevels = 1:6))


# -

tab1 = FreqTables.freqtable(dba.Y,A.za_pred)
tab2 = FreqTables.freqtable(dbb.Z,A.yb_pred)
@show tab2
#accZ=(tab1[1,1]+tab1[2,2]+tab1[3,3]+tab1[4,4]+tab1[4,5])/sum(tab1)
#accY=(tab2[1,1]+tab2[2,2]+tab2[3,3]+tab2[4,4]+tab2[5,4]+tab2[6,4])/sum(tab2)
@show tab2
accZ=(tab1[1,1]+tab1[2,2]+tab1[3,3]+tab1[4,4]+tab1[4,5])/sum(tab1)


# OT-between à la main

# +
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

colnames = names(data, r"^X")
XA = transpose(Matrix{Float32}(dba[!, colnames]))
XB = transpose(Matrix{Float32}(dbb[!, colnames]))

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

# -

# Transport des covariables - choix du rho optimal

# +
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))
XA1=dba[:,1]
XA2=dba[:,2].+1
XA3=dba[:,3].+1
XA4=dba[:,4]
XA5=dba[:,5]
XA2f = Flux.onehotbatch(XA2, 1:2)
XA3f = Flux.onehotbatch(XA3, 1:2)
XA5f = Flux.onehotbatch(XA5, 1:8)
YA = Flux.onehotbatch(dba.Y, Ylevels)
ZB = Flux.onehotbatch(dbb.Z, Zlevels)

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

nA = size(dba, 1)
nB = size(dbb, 1)

wa = ones(Float32, nA) ./ nA
wb = ones(Float32, nB) ./ nB
print(size(XA1))

# -

@show C

XB1=dbb[:,1]
XB2=dbb[:,2].+1
XB3=dbb[:,3].+1
XB4=dbb[:,4]
XB5=dbb[:,5]
XB2f = Flux.onehotbatch(XB2, 1:2)
XB3f = Flux.onehotbatch(XB3, 1:2)
XB5f = Flux.onehotbatch(XB5, 1:8)

# Choix du paramètre rho optimal

# +
# listes de valeurs
reg_m1_list = [0.01, 0.1,0.5, 1.0]
reg_m2_list = [0.01, 0.1, 0.5,1.0]
#reg_m1_list = [0.01]
#reg_m2_list = [0.01]
reg= 0.001
results = []   # vecteur vide pour stocker les tuples (reg_m1, reg_m2, cost)
XBpred1 = Vector{Float64}[]
XBpred2 = Matrix{Float64}[]
XBpred3 = Matrix{Float64}[]
XBpred4 = Vector{Float64}[]
XBpred5 = Matrix{Float64}[]
XBpred = Matrix{Float64}[]
XB = DataFrame(
    X1 = XB1,
    X2 = XB2,
    X3 = XB3,
    X4 = XB4,
    X5 = XB5
)
for r1 in reg_m1_list
    for r2 in reg_m2_list
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
        XBpred = DataFrame(
                X1 = XBpred1,
                X2 = XBpred2,
                X3 = XBpred3,
                X4 = XBpred4,
                X5 = XBpred5
                )
        cost=0
		for i in 1:nB
            cost += Distances.evaluate(dist, XB[i,:], XBpred[i,:])
        end
        push!(results, (reg_m1=r1, reg_m2=r2, cost=cost))
    end
end
best = findmin([r.cost for r in results])  # retourne (valeur_min, index)
best_value, idx = best
best_params = results[idx]

println("Meilleur coût = ", best_value)
println("Paramètres optimaux = ", best_params)

# -

best_params

@show XBpred

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

# +

XApred1 = (1 ./row_sumsB) .* vec(XB1' * G')
XApred2f = (XB2f* G') .* (1 ./row_sumsB)'
XApred2 = Flux.onecold(XApred2f)
XApred3f = (XB3f* G') .* (1 ./row_sumsB)'
XApred3 = Flux.onecold(XApred3f)
XApred4 =(1 ./row_sumsB) .* vec(XB4' * G')
XApred5f = (XB5f* G') .* (1 ./row_sumsB)'
XApred5 = Flux.onecold(XApred5f)
# -

XAt = hcat(XA1,XA2f',XA3f',XA4,XA5f')
XBpredt = hcat(XBpred1,XBpred2f',XBpred3f',XBpred4,XBpred5f')
XA=XAt'
XBpred=XBpredt'
XBt = hcat(XB1,XB2f',XB3f',XB4,XB5f')
XApredt = hcat(XApred1,XApred2f',XApred3f',XApred4,XApred5f')
XB=XBt'
XApred=XApredt'

# +
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


# +
train!(modelXYA, XApred, YA)
train!(modelXZB, XBpred, ZB)
ZApred = modelXZB(XA)
YBpred = modelXYA(XB)

ybpred=Flux.onecold(YBpred)
zapred= Flux.onecold(ZApred)
# -

using Pkg
using FreqTables
tab1 = FreqTables.freqtable(dba.Y,zapred)
tab2 = FreqTables.freqtable(dbb.Z,ybpred)
@show tab2
#accZ=(tab1[2,1]+tab1[3,2]+tab1[4,3]+tab1[4,4])/sum(tab1)
accY=(tab2[1,1]+tab2[2,2]+tab2[3,3]+tab2[4,4]+tab2[5,4]+tab2[6,4])/sum(tab2)


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

# +
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

# +
wa = ones(Float32, nA) ./ nA
wb = ones(Float32, nB) ./ nB

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
        #row_sumsB = vec(sum(G, dims=2))
        #col_sumsA = vec(sum(G, dims=1))
        #YB .=  (YA * G).* (1 ./col_sumsA)'
        #ZA .=  (ZB * G').* (1 ./row_sumsB)'
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



# +
tab1 = FreqTables.freqtable(dba.Y,zapred)
tab2 = FreqTables.freqtable(dbb.Z,ybpred)
@show tab1

#accZ=(tab1[2,1]+tab1[3,2]+tab1[4,3]+tab1[4,4])/sum(tab1)
#accY=(tab2[1,1]+tab2[2,2]+tab2[3,3]+tab2[4,4]+tab2[5,4]+tab2[6,4])/sum(tab2)

#accY=(tab2[1,1]+tab2[2,2]+tab2[4,3]+tab2[5,3]+tab2[6,3])/sum(tab2)


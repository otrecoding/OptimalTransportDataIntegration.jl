# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.10.3
#     language: julia
#     name: julia-1.10
# ---

import Pkg
Pkg.add(; url="https://github.com/otrecoding/OTRecod.jl.git")

# +
using OptimalTransportDataIntegration

params = DataParameters(nA = 1000, nB = 500)

data = generate_xcat_ycat(params)

# +
"""
One hot encoding of the array X, number of colums is equal to n

levels in array must be in [0, n[
"""
function onehot(x :: AbstractArray; n = 4)
    res = Vector{Int}[]  # Création d'un vecteur de vecteurs d'entiers
    levels = 0:n
    for col in eachcol(x)
        for level in levels
            push!(res, col .== level)
        end
    end
    return hcat(res...) # Transformation en matrice du vecteur de vecteurs
end

onehot(data.Y)
# -

"""
    optimal_modality(value, loss, weight)

- `values`: vector of possible values
- `weight`: vector of weights 
- `loss`: matrix of size len(weight) * len(values)
- returns an argmin over value in Values of the scalar product <Loss[value,],Weight> 
"""   
function optimal_modality(value, loss, weight)
    
    cost = Float64[]
    for j in eachindex(value)
        s = 0
        for i in eachindex(weight)
            s += loss[i,j] * weight[i]
        end
        push!(cost, s)
    end
    return values[argmin(cost)]
end

function onehot(X, n=4):
    "One hot encoding of the matrix X, number of colums is equal to n"
    return pd.get_dummies(X, dtype=np.int32).reindex(columns=range(n)).fillna(0).values

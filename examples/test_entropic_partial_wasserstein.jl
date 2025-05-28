
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

import OptimalTransportDataIntegration: entropic_partial_wasserstein
import CSV
using DataFrames
import PythonOT
import Distances: pairwise, Hamming

T = Int

csv_file = joinpath("dataset.csv")

data = CSV.read(csv_file, DataFrame)

Ylevels = 1:4
Zlevels = 1:3

X = Matrix{Int}(one_hot_encoder(data[!, [:X1, :X2, :X3]]))
Xlevels = sort(unique(eachrow(X)))
Y = Vector{T}(data.Y)
Z = Vector{T}(data.Z)

base = data.database

indA = findall(base .== 1)
indB = findall(base .== 2)


YA = view(Y, indA)
YB = view(Y, indB)
ZA = view(Z, indA)
ZB = view(Z, indB)

XA = view(X, indA, :)
XB = view(X, indB, :)

nA = length(indA)
nB = length(indB)

# Optimal Transport

C = pairwise(Hamming(), XA, XB; dims = 1)

wa = vec([sum(indXA[x][YA[indXA[x]] .== y]) for y in Ylevels, x in eachindex(indXA)])
wb = vec([sum(indXB[x][ZB[indXB[x]] .== z]) for z in Zlevels, x in eachindex(indXB)])

wa2 = filter(>(0), wa) ./ nA
wb2 = filter(>(0), wb) ./ nB

nx = size(X, 2) ## Nb modalités x 

XYA2 = Vector{T}[]
XZB2 = Vector{T}[]
for (i, (y, x)) in enumerate(product(Ylevels, Xlevels))
    wa[i] > 0 && push!(XYA2, [x...; y])
end
for (i, (z, x)) in enumerate(product(Zlevels, Xlevels))
    wb[i] > 0 && push!(XZB2, [x...; z])
end


XA_hot = stack([v[1:nx] for v in XYA2], dims = 1)
XB_hot = stack([v[1:nx] for v in XZB2], dims = 1)

## Optimal Transport

C = pairwise(Hamming(), XA_hot, XB_hot; dims = 1)

G = entropic_partial_wasserstein(wa2, wb2, C, 0.1)

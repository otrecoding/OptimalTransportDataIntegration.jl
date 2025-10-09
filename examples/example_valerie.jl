# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Julia 1.11.6
#     language: julia
#     name: julia-1.11
# ---

# %%
using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf
using DataFrames
using Flux
using Plots

# %%
params = DataParameters(nA = 1000,
    nB = 1000,
    mA = [0.0],
    mB = [4.0],
    covA = ones(1,1),
    covB = ones(1,1),
    aA = [1.0],
    aB = [1.0],
    r2 = 0.9)
 
rng = ContinuousDataGenerator(params; scenario = 2)
data = generate(rng)

# %%
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))


cols = names(dba, r"^X")   

XA = transpose(Matrix(dba[:, cols]))
XB = transpose(Matrix(dbb[:, cols]))

XAi = round.(XA)
XBj = round.(XB)

# %%
XA
XB

# %%
plot(
    histogram(vec(XA), bins=20, xlabel="XA", ylabel="Fréquence", title="Histogramme de XA"),
    histogram(vec(XB), bins=20, xlabel="XB", ylabel="Fréquence", title="Histogramme de XB"),
    layout = (1,2)   # 1 ligne, 2 colonnes
)


# %%
Ylevels = 1:4
Zlevels = 1:3
YA = Flux.onehotbatch(dba.Y, Ylevels)
ZB = Flux.onehotbatch(dbb.Z, Zlevels)



# %%
show(dba.Y)
show(YA)

# %%
show(dbb.Z)
show(ZB)

# %%
using Distances
nA = size(dba, 1)
nB = size(dbb, 1)

wa = ones(nA) ./ nA
wb = ones(nB) ./ nB

C0 = pairwise(Euclidean(), XA, XB, dims = 2)

C = C0 #./ maximum(C0)
C2=C.^2
display(C)
display(wa)
display(wb)

# %%
using PythonOT

reg = 0.0
reg_m1 = 0.0
reg_m2 = 0.0
G = ones(length(wa), length(wb))

#if reg > 0
#        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
#else
G .= PythonOT.emd(wa, wb, C2)
#end
show(size(ZB))
show(size(G'))
show(size(YA))
ZApred =  nA .* ZB * G'
YBpred =nB .* YA * G

# %%
display(XAi)
display(sort(vec(XAi)))
io = sortperm(vec(XAi))
display(sort(vec(YA[io])))
display(sort(vec(dba.Z[io])))
display(io)
display(sort(vec(XBj)))
jo = sortperm(vec(XBj)) 
display(jo)
display(sort(vec(ZB[io])))
display(sort(vec(dbb.Y[io])))
A = G[io,:]
display(A)
B=A[:,jo]
display(B)

# %%
pos_idx = findall(x -> x > 0, G)
display(pos_idx)

# %%
XAv=vec(XA)
XBv=vec(XB)
show(pos_idx[1])
show(XAv[pos_idx[1][1]])
show(XBv[pos_idx[1][2]])

show(XAv[pos_idx[2][1]])
show(XBv[pos_idx[2][2]])


# %%
using Flux
display(ZB)
display(dbb.Z)
display(G)
show(Flux.onecold(ZApred))
show(accuracy(Flux.onecold(ZApred), dba.Z))


# %%
show(accuracy(Flux.onecold(ZApred), dba.Z))
show(accuracy(Flux.onecold(YBpred), dbb.Y))

# %%
display(G)
display(G')
display(dbb.Z)
display(dba.Z)

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

using DataFrames
using OptimalTransportDataIntegration
using StatsBase
using XLSX

data = DataFrame(XLSX.readtable("data.xlsx", 1))
data.X1 = convert(Vector{Float32}, data.X1)
data.X2 = convert(Vector{Int}, data.X2)
data.X3 = convert(Vector{Int}, data.X3)
data.X4 = convert(Vector{Float32}, data.X4)
data.X5 = convert(Vector{Int}, parse.(Int, data.X5))
data.Y = convert(Vector{Union{Missing, Int}}, data.Y)
data.Z = convert(Vector{Union{Missing, Int}}, data.Z)
data

xcols = names(data, r"^X")
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))
XA = dba[:, xcols]
XB = dbb[:, xcols]

# +

digitize_int(XA, XB, [:X1, :X4])
digitize_median(XA, XB, [:X1, :X4])
# -

@show cols = names(data, r"^X")
Y = Vector{Union{Missing, Int}}(data.Y)
Z = Vector{Union{Missing, Int}}(data.Z)
XA = transpose(Matrix{Float32}(dba[:, cols]))
XB = transpose(Matrix{Float32}(dbb[:, cols]))


# +
using CategoricalArrays
cols = names(data, r"^X")
for name in ["X2", "X3", "X5"]
    lev = union(unique(dba[!, name]), unique(dbb[!, name]))
    dba[!, name] = categorical(dba[!, name], levels = lev)
    dbb[!, name] = categorical(dbb[!, name], levels = lev)
end


# -

dba.X4

A = dba[:, cols]
B = dbb[:, cols]
C = vcat(A, B)
dist = GowerDF2([:X1, :X4], [:X2, :X3, :X5], C)


D = pairwise_gower(dist, A, B)

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:light,ipynb
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Julia 1.11.1
#     language: julia
#     name: julia-1.11
# ---

# +
using OptimalTransportDataIntegration

params = DataParameters(nA = 1000, nB = 1000, mB = [2, 0, 0], eps = 0.0, p = 0.2)
data = generate_xcat_ycat(params)
@show sort(unique(data.Y)), sort(unique(data.Z))
@time est = unbalanced_modality(data)
println(est)

params = DataParameters(nA = 1000, nB = 1000, mB = [0, 0, 0], eps = 0.0, p = 0.2)
data = generate_xcat_ycat(params)
@show sort(unique(data.Y)), sort(unique(data.Z))
@time est = unbalanced_modality(data)
println(est)

data = CSV.read(joinpath(@__DIR__, "../test/data_good.csv"), DataFrame)
@show sort(unique(data.Y)), sort(unique(data.Z))
@time est = unbalanced_modality(data)
println(est)

data = CSV.read(joinpath(@__DIR__, "../test/data_bad.csv"), DataFrame)
@show sort(unique(data.Y)), sort(unique(data.Z))
@time est = unbalanced_modality(data)
println(est)

params = DataParameters(nA = 1000, nB = 500)
data = generate_xcat_ycat(params)
@show sort(unique(data.Y)), sort(unique(data.Z))
@time est = unbalanced_modality(data)
println(est)

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
using CSV
using DataFrames
using OptimalTransportDataIntegration

params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], eps = 0.0, p = 0.2)
data = generate_xcat_ycat(params)

est = otrecod(data, OTjoint())
println("OT-r : $est")

est = otrecod(data, UnbalancedModality(iterations = 10))
println("OTE-r : $est")

est = otrecod(data, UnbalancedModality(reg = 0.1, reg_m = 0.0))
println("OTE : $est")

params = DataParameters(nA = 1000, nB = 1000, mB = [2, 0, 0], eps = 0.0, p = 0.2)
data = generate_xcat_ycat(params)

est = otrecod(data, OTjoint())
println("OT-r : $est")
est = otrecod(data, UnbalancedModality(iterations = 10))
println("OTE-r : $est")

est = otrecod(data, UnbalancedModality(reg_m = 0.0))
println("OTE : $est")

data = CSV.read(joinpath(@__DIR__, "../test/data_good.csv"), DataFrame)
est = otrecod(data, OTjoint())
println("OT-r : $est")
est = otrecod(data, UnbalancedModality(iterations = 10))
println("OTE-r : $est")

data = CSV.read(joinpath(@__DIR__, "../test/data_bad.csv"), DataFrame)
est = otrecod(data, OTjoint())
println("OT-r : $est")
est = otrecod(data, UnbalancedModality(iterations = 10))
println("OTE-r : $est")

params = DataParameters(nA = 1000, nB = 500)
data = generate_xcat_ycat(params)
est = otrecod(data, OTjoint())
println("OT-r : $est")
est = otrecod(data, UnbalancedModality(iterations = 10))
println("OTE-r : $est")

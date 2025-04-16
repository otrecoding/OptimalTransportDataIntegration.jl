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

params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], p = 0.2)
rng = PDataGenerator(params)
data = generate_data(rng)

result = otrecod(data, JointOTWithinBase())
est = accuracy(result)

print(" OT-r : $est")
est = otrecod(data, JointOTBetweenBases(reg = 0.1, reg_m1 = 0.0, reg_m2 = 0.0))
est = accuracy(result)
print(" OTE : $est")
est = otrecod(data, JointOTBetweenBases(iterations = 10))
est = accuracy(result)
print(" OTE-r : $est")
est = otrecod(data, SimpleLearning())
est = accuracy(result)
println(" SL : $est")

params = DataParameters(nA = 1000, nB = 1000, mB = [2, 0, 0], p = 0.2)
rng = PDataGenerator(params)
data = generate_data(rng)

result = otrecod(data, JointOTWithinBase())
est = accuracy(result)
print(" OT-r : $est")
result = otrecod(data, JointOTBetweenBases(reg = 0.1, reg_m1 = 0.0, reg_m2 = 0.0))
est = accuracy(result)
print(" OTE : $est")
result = otrecod(data, JointOTBetweenBases(iterations = 10))
est = accuracy(result)

print(" OTE-r : $est")
result = otrecod(data, SimpleLearning())
est = accuracy(result)
println(" SL : $est")

data = CSV.read(joinpath(@__DIR__, "../test/data_good.csv"), DataFrame)
result = otrecod(data, JointOTWithinBase())
est = accuracy(result)
print(" OT-r : $est")
result = otrecod(data, JointOTBetweenBases(reg = 0.1, reg_m1 = 0.0, reg_m2 = 0.0))
est = accuracy(result)
print(" OTE : $est")
result = otrecod(data, JointOTBetweenBases(iterations = 10))
est = accuracy(result)
print(" OTE-r : $est")
result = otrecod(data, SimpleLearning())
est = accuracy(result)
println(" SL : $est")

data = CSV.read(joinpath(@__DIR__, "../test/data_bad.csv"), DataFrame)
result = otrecod(data, JointOTWithinBase())
est = accuracy(result)
print(" OT-r : $est")
result = otrecod(data, JointOTBetweenBases(reg = 0.1, reg_m1 = 0.0, reg_m2 = 0.0))
est = accuracy(result)
print(" OTE : $est")
result = otrecod(data, JointOTBetweenBases(iterations = 10))
est = accuracy(result)
print(" OTE-r : $est")
result = otrecod(data, SimpleLearning())
est = accuracy(result)
println(" SL : $est")

params = DataParameters(nA = 1000, nB = 1000)
rng = PDataGenerator(params)
data = generate_data(rng)

result = otrecod(data, JointOTWithinBase())
est = accuracy(result)
print(" OT-r : $est")
result = otrecod(data, JointOTBetweenBases(reg = 0.1, reg_m1 = 0.0, reg_m2 = 0.0))
est = accuracy(result)
print(" OTE : $est")
result = otrecod(data, JointOTBetweenBases(iterations = 10))
est = accuracy(result)
print(" OTE-r : $est")
result = otrecod(data, SimpleLearning())
est = accuracy(result)
println(" SL : $est")

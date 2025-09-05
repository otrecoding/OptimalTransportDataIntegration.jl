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

for mB in [[1, 0, 0], [5, 5, 5], [10, 10, 10]]

    println("mB = $mB .......")
    params = DataParameters(mB = mB)
    rng = DiscreteDataGenerator(params)
    data = generate(rng)
    result = otrecod(data, JointOTWithinBase())
    est = accuracy(result)
    print(" within-r : $est")
    est = otrecod(data, JointOTBetweenBases(reg = 0.1, reg_m1 = 0.0, reg_m2 = 0.0))
    est = accuracy(result)
    print(" between : $est")
    est = otrecod(data, JointOTBetweenBases())
    est = accuracy(result)
    print(" between-r : $est")
    est = otrecod(data, SimpleLearning())
    est = accuracy(result)
    println(" sl : $est")

end

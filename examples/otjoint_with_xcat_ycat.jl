# ---
# jupyter:
#   jupytext:
#     formats: jl,ipynb
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

params = DataParameters(nA = 1000, nB = 1000)

data = generate_xcat_ycat(params)

# +
import OptimalTransportDataIntegration: Instance, ot_joint
import Distances: Hamming

X = Matrix(data[!, ["X1", "X2", "X3"]])
Y = Vector(data.Y)
Z = Vector(data.Z)
database = data.database

dist_choice = Hamming()

Ylevels = 1:4
Zlevels = 1:3
    
instance = Instance( database, X, Y, Ylevels, Z, Zlevels, dist_choice)
    
lambda_reg = 0.392
maxrelax = 0.714
percent_closest = 0.2
    
sol = ot_joint(instance, maxrelax, lambda_reg, percent_closest)
compute_pred_error!(sol, instance, false)
# -


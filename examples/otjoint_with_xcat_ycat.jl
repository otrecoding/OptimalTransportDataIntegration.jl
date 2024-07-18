# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Julia 1.10.4
#     language: julia
#     name: julia-1.10
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
    
instance = Instance( database, X, Y, Z, dist_choice)
    
lambda_reg = 0.392
maxrelax = 0.714
percent_closest = 0.2
    
sol = ot_joint(instance, maxrelax, lambda_reg, percent_closest)
compute_pred_error!(sol, instance, false)
# -


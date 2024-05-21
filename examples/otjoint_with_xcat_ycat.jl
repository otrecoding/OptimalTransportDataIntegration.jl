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
import OTRecod: Instance, ot_joint
import Distances: Hamming

X = Matrix(data[!, ["X1", "X2", "X3"]])
Y = Vector(data.Y)
Z = Vector(data.Z)
database = data.database

dist_choice = Hamming()
    
instance = OTRecod.Instance( database, X, Y, Z, dist_choice)
    
lambda_reg = 0.392
maxrelax = 0.714
percent_closest = 0.2
    
sol = OTRecod.ot_joint(instance, maxrelax, lambda_reg, percent_closest)
OTRecod.compute_pred_error!(sol, instance, false)
# -


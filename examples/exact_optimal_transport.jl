# -*- coding: utf-8 -*-
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

import Pkg;
Pkg.add(["ExactOptimalTransport", "Tulip"]);

# +
using ExactOptimalTransport
using Distances
import Tulip
import PythonOT

# uniform histograms
μ = fill(1 / 10, 10)
ν = fill(1 / 20, 20)

# random cost matrix
C = pairwise(SqEuclidean(), rand(1, 10), rand(1, 20); dims = 2)

# compute optimal transport map with Tulip
lp = Tulip.Optimizer()

maximum(abs.(emd(μ, ν, C, lp) .- PythonOT.emd(μ, ν, C)))
# -

emd2(μ, ν, C, lp) - PythonOT.emd2(μ, ν, C)

emd2(μ, ν, C, lp) .- sum(C .* PythonOT.emd(μ, ν, C))

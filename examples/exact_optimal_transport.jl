# -*- coding: utf-8 -*-
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
#     display_name: Julia 1.10.3
#     language: julia
#     name: julia-1.10
# ---

# +
using ExactOptimalTransport
using Distances
import Tulip
import PythonOT

# uniform histograms
μ = fill(1/10, 10)
ν = fill(1/20, 20)

# random cost matrix
C = pairwise(SqEuclidean(), rand(1, 10), rand(1, 20); dims=2)

# compute optimal transport map with Tulip
lp = Tulip.Optimizer()

maximum(abs.(emd(μ, ν, C, lp) .- PythonOT.emd(μ, ν, C)))
# -

emd2(μ, ν, C, lp) -  PythonOT.emd2(μ, ν, C)

emd2(μ, ν, C, lp) .- sum(C .* PythonOT.emd(μ, ν, C))



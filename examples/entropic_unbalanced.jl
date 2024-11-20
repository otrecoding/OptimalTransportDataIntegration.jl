# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl
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

# # 2D examples of exact and entropic unbalanced optimal transport
#
# https://pythonot.github.io/master/auto_examples/unbalanced-partial/plot_unbalanced_OT.html

import Pkg;
Pkg.add("PyPlot");

# +
using LinearAlgebra
using Distances
import PythonOT
import PyPlot

"""
    make_2D_samples_gauss(n, m, sigma)

Return `n` samples drawn from 2D gaussian ``\\mathcal{N}(m, \\sigma)``
"""
make_2D_samples_gauss(n, m, sigma) = randn(n, 2) * sqrt(sigma) .+ m'


# +
n = 40  # nb samples

mu_s = [-1, -1]
cov_s = [1 0; 0 1]
# -

mu_t = [4, 4]
cov_t = [1 -0.8; -0.8 1]

xs = make_2D_samples_gauss(n, mu_s, cov_s)
xt = make_2D_samples_gauss(n, mu_t, cov_t)

# +
n_noise = 10

xs = vcat(xs, rand(n_noise, 2) .- 4)
xt = vcat(xt, rand(n_noise, 2) .+ 6)

# +
n = n + n_noise

a, b = ones(n) ./ n, ones(n) ./ n  # uniform distribution on samples

# +

# loss matrix
M = pairwise(SqEuclidean(), xs, xt, dims = 1)
M ./= maximum(M)
# -

# ## Compute kl-regularized UOT

# +
reg_m_kl = 0.05
reg_m_l2 = 5

P = PythonOT.mm_unbalanced(a, b, M, reg_m_kl, div = "kl")
# -

# ## Plot the results

if sum(P) > 0
    P .= P ./ maximum(P)
end
for i in axes(P, 1), j in axes(P, 2)
    if P[i, j] > 0
        PyPlot.plot(
            [xs[i, 1], xt[j, 1]],
            [xs[i, 2], xt[j, 2]],
            color = "C2",
            alpha = P[i, j] * 0.3,
        )
    end
end
PyPlot.scatter(xs[:, 1], xs[:, 2], c = "C0")
PyPlot.scatter(xt[:, 1], xt[:, 2], c = "C1")
PyPlot.scatter(xs[:, 1], xs[:, 2], c = "C0", s = vec(sum(P, dims = 2)))
PyPlot.scatter(xt[:, 1], xt[:, 2], c = "C1", s = vec(sum(P, dims = 1)))

PyPlot.imshow(P, cmap = "jet")

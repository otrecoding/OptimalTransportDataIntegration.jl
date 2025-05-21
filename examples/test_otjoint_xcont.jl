# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,jl
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
# ---

using OptimalTransportDataIntegration
using DataFrames
using Distances
using Statistics


digitize(x, bins) = searchsortedlast.(Ref(bins), x)

function categorize_using_quartile(data)

    XA = subset(data, :database => x -> x .== 1.0)
    XB = subset(data, :database => x -> x .== 2.0)

    b1 = quantile(XA.X1, [0.25, 0.5, 0.75])
    bins11 = vcat(-Inf, b1, +Inf)

    X11 = digitize(XA.X1, bins11)
    X21 = digitize(XB.X1, bins11)

    b1 = quantile(XA.X2, [0.25, 0.5, 0.75])
    bins12 = vcat(-Inf, b1, +Inf)

    X12 = digitize(XA.X2, bins12)
    X22 = digitize(XB.X2, bins12)

    b1 = quantile(XA.X3, [0.25, 0.5, 0.75])
    bins13 = vcat(-Inf, b1, +Inf)

    X13 = digitize(XA.X3, bins13)
    X23 = digitize(XB.X3, bins13)

    X1 = vcat(X11, X21) .- 1
    X2 = vcat(X12, X22) .- 1
    X3 = vcat(X13, X23) .- 1

    hcat(X1, X2, X3)

end

params = DataParameters()
rng = DataGenerator(params, discrete = false)
data = generate(rng)


X = categorize_using_quartile(data)
Y = Vector(data.Y)
Z = Vector(data.Z)

Ylevels = 1:4
Zlevels = 1:3

database = data.database
dist_choice = Euclidean()

instance = Instance(database, X, Y, Ylevels, Z, Zlevels, dist_choice)

lambda = 0.1
alpha = 0.1
percent_closest = 0.2

sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

YB, ZA = compute_pred_error!(sol, instance, false)

println(accuracy(data, YB, ZA))

println(accuracy(otrecod(data, JointOTWithinBase(distance=Cityblock()))))
println(accuracy(otrecod(data, JointOTWithinBase(distance=Hamming()))))
println(accuracy(otrecod(data, JointOTWithinBase(distance=Euclidean()))))
println(accuracy(otrecod(data, JointOTBetweenBases())))
println(accuracy(otrecod(data, SimpleLearning())))

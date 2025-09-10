using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf

params = DataParameters(mA = [0.0], 
                        mB = [2.0],
                        covA = ones(1,1),
                        covB = ones(1,1),
                        aA = [1.0],
                        aB = [1.0],
                        r2 = 0.6)
 
rng = ContinuousDataGenerator(params; scenario = 2)
data = generate(rng)

#= sol = otrecod(data, SimpleLearning())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTWithinBase())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTBetweenBases())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTBetweenBasesWithoutYZ())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTBetweenBasesRefJDOT())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTBetweenBasesRefOTDAx())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTBetweenBasesRefOTDAyz())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true)
sol = otrecod(data, JointOTBetweenBasesRefOTDAyzPred())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
@show confusion_matrix(sol.za_pred, sol.za_true) =#

using DataFrames
using Flux
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))


cols = names(dba, r"^X")   

XA = transpose(Matrix(dba[:, cols]))
XB = transpose(Matrix(dbb[:, cols]))

YA = Flux.onehotbatch(dba.Y, Ylevels)
ZB = Flux.onehotbatch(dbb.Z, Zlevels)

    nA = size(dba, 1)
    nB = size(dbb, 1)

    wa = ones(nA) ./ nA
    wb = ones(nB) ./ nB

    C0 = pairwise(Euclidean(), XA, XB, dims = 2)

    C = C0 ./ maximum(C0)

    G = ones(length(wa), length(wb))

    if reg > 0
        G .= PythonOT.mm_unbalanced(wa, wb, C, (reg_m1, reg_m2); reg = reg, div = "kl")
    else
        G .= PythonOT.emd(wa, wb, C)
    end

    ZApred = Flux.softmax(nA .* ZB * G')
    YBpred = Flux.softmax(nB .* YA * G)
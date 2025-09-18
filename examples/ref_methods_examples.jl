using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf

params = DataParameters(nA = 1000,
    nB = 1000,
    mA = [0.0],
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



@show accuracy(otrecod(data, SimpleLearning()))

@show accuracy(otrecod(data, JointOTWithinBase()))

@show accuracy(otrecod(data, JointOTBetweenBaseswithpred(reg = 0.0)))

@show accuracy(otrecod(data, JointOTBetweenBasesWithoutYZ(reg = 0.0)))

@show accuracy(otrecod(data, JointOTBetweenBasesRefJDOT(reg = 0.0)))

@show accuracy( otrecod(data, JointOTBetweenBasesRefOTDAx(reg = 0.0)))

@show accuracy(otrecod(data, JointOTBetweenBasesRefOTDAyz(reg = 0.0)))

@show accuracy( otrecod(data, JointOTBetweenBasesRefOTDAyzPred(reg = 0.0)))



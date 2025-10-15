using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf
using TimerOutputs

const to = TimerOutput()

params = DataParameters(nA = 1000,
    nB = 1000,
    mA = [0.0,0.0,0.0],
    mB = [1.0,1.0,0.0],
    covA = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
    covB = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
    aA = [1.0,1.0,1.0],
    aB = [1.0,1.0,1.0],
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



@timeit to "Simple Learning" est = accuracy(otrecod(data, SimpleLearning()))
println("simple learning : $est")

# @timeit to "Within" accuracy(otrecod(data, JointOTWithinBase()))

@timeit to "Between with predictors"  est = accuracy(otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0)))
println("between with predictors $est")

@timeit to "Between without outcomes" est = accuracy(otrecod(data, JointOTBetweenBasesWithoutYZ(reg = 0.0)))
println("between without outcomes $est")

@timeit to "JDOT" est = accuracy(otrecod(data, JointOTBetweenBasesRefJDOT(reg = 0.0)))
println("jdot $est")

@timeit to "OTDA covariates" est = accuracy( otrecod(data, JointOTBetweenBasesRefOTDAx(reg = 0.0)))
println("otda covariates $est")

@timeit to "OTDA outcomes" est = accuracy(otrecod(data, JointOTBetweenBasesRefOTDAyz(reg = 0.0)))
println("otda outcomes $est")

@timeit to "OTDA outcomes with predictors" est = accuracy( otrecod(data, JointOTBetweenBasesRefOTDAyzPred(reg = 0.0)))
println("otda with predictors $est")


@show(to)



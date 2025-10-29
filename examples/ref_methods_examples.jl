using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf
using TimerOutputs

const to = TimerOutput()

params = DataParameters(
    nA = 1000,
    nB = 1000,
    mA = [0.0, 0.0, 0.0],
    mB = [1.0, 1.0, 0.0],
    covA = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
    covB = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
    aA = [1.0, 1.0, 1.0],
    aB = [1.0, 1.0, 1.0],
    r2 = 0.6
)

rng = ContinuousDataGenerator(params; scenario = 2)
data = generate(rng)

@timeit to "SimpleLearning" est = accuracy(otrecod(data, SimpleLearning()))
println("SimpleLearning : $est")

@timeit to "JointOTWithinBase" accuracy(otrecod(data, JointOTWithinBase()))
println("JointOTWithinBase : $est")

@timeit to "JointOTBetweenBasesWithPredictors"  est = accuracy(otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0)))
println("JointOTBetweenBasesWithPredictors $est")

@timeit to "JointOTBetweenBasesWithoutOutcomes" est = accuracy(otrecod(data, JointOTBetweenBasesWithoutOutcomes(reg = 0.0)))
println("JointOTBetweenBasesWithoutOutcomes $est")

@timeit to "JointOTBetweenBasesJDOT" est = accuracy(otrecod(data, JointOTBetweenBasesJDOT(reg = 0.0)))
println("JointOTBetweenBasesJDOT $est")

@timeit to "JointOTDABetweenBasesCovariables" est = accuracy(otrecod(data, JointOTDABetweenBasesCovariables(reg = 0.0)))
println("JointOTDABetweenBasesCovariables $est")

@timeit to "JointOTDABetweenBasesOutcomes" est = accuracy(otrecod(data, JointOTDABetweenBasesOutcomes(reg = 0.0)))
println("JointOTDABetweenBasesOutcomes $est")

@timeit to "JointOTDABetweenBasesOutcomesWithPredictors" est = accuracy(otrecod(data, JointOTDABetweenBasesOutcomesWithPredictors(reg = 0.0)))
println("JointOTDABetweenBasesOutcomesWithPredictors $est")

@show(to)

using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf

params = DataParameters(
    nA = 1000,
    nB = 1000,
    mA = [0.0],
    mB = [2.0],
    covA = ones(1,1),
    covB = ones(1,1),
    aA = [1.0],
    aB = [1.0],
    r2 = 0.9,
    pA = [[0.5, 0.5]],
    pB= [[ 0.8, 0.2]]  )
 
rng = DiscreteDataGenerator(params; scenario = 2)
data = generate(rng)



@show accuracy(otrecod(data, SimpleLearning()))

@show accuracy(otrecod(data, JointOTWithinBase()))

@show accuracy(otrecod(data, JointOTBetweenBaseswithpred(reg = 0.0)))

@show accuracy(otrecod(data, JointOTBetweenBases(reg_m1=0)))

@show accuracy(otrecod(data, JointOTBetweenBasesc(reg_m1=0)))
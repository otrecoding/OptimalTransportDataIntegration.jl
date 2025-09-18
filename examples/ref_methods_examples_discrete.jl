using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf

params = DataParameters()
 
rng = DiscreteDataGenerator(params; scenario = 2)
data = generate(rng)



@show accuracy(otrecod(data, SimpleLearning()))

@show accuracy(otrecod(data, JointOTWithinBase()))

@show accuracy(otrecod(data, JointOTBetweenBases(reg = 0.0)))


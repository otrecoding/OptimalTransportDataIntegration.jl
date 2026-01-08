using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf


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

rng = ContinuousDataGenerator(params; scenario = 1)
data = generate(rng)

est = accuracy(otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.1)))
@show est
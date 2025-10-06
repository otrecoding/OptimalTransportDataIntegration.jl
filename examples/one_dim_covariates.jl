using OptimalTransportDataIntegration

params = DataParameters(
    mA = [0.0],
    mB = [0.0],
    covA = ones(1, 1),
    covB = ones(1, 1),
    aA = [1.0],
    aB = [1.0]
)

rng = DiscreteDataGenerator(params)
data = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))
@show accuracy(otrecod(data, JointOTWithinBase()))
@show accuracy(otrecod(data, JointOTBetweenBases()))

rng = ContinuousDataGenerator(params)
ata = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))
@show accuracy(otrecod(data, JointOTWithinBase()))
@show accuracy(otrecod(data, JointOTBetweenBases()))

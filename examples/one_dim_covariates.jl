using OptimalTransportDataIntegration
using Test

params = DataParameters()  # X1, X2, X3

rng = DiscreteDataGenerator(params)
data = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))

rng = ContinuousDataGenerator(params)
data = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))

# Only X1
params = DataParameters(mA = [0.0], 
                        mB = [0.0],
                        covA = ones(1,1),
                        covB = ones(1,1),
                        aA = [1.0],
                        aB = [1.0])
 
rng = DiscreteDataGenerator(params)
data = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))
 
rng = ContinuousDataGenerator(params)
ata = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))

# X1, X2
params = DataParameters(mA = [0.0, 0.0], mB = [0.0, 0.0],
                        covA = [1.0 0.2; 0.2 1.0],
                        covB = [1.0 0.2; 0.2 1.0],
                        pA = [[0.5, 0.5], [0.333, 0.334, 0.333]],
                        pB = [[0.7, 0.3], [0.333, 0.334, 0.333]],
                        aA = [1.0, 1.0],
                        aB = [1.0, 1.0])

rng = DiscreteDataGenerator(params)
data = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))
 
rng = ContinuousDataGenerator(params)
ata = generate(rng)
@show accuracy(otrecod(data, SimpleLearning()))

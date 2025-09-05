using OptimalTransportDataIntegration
using Test

params = DataParameters()
rng = DiscreteDataGenerator(params)
@show data = generate(rng)

@test all(names(data) .== [ "X1", "X2", "X3", "Y", "Z", "database"])

params = DataParameters(mA = [0.0, 0.0], mB = [0.0, 0.0],
                        covA = [1.0 0.2; 0.2 1.0],
                        covB = [1.0 0.2; 0.2 1.0],
                        pxc = [[0.5, 0.5], [0.333, 0.334, 0.333]],
                        aA = [1.0, 1.0, 1.5],
                        aB = [1.0, 1.0, 1.5])

rng = DiscreteDataGenerator(params)
@show data = generate(rng)

params = DataParameters(mA = [0.0], mB = [0.0],
                        covA = ones(1,1),
                        covB = ones(1,1),
                        pxc = [[0.5, 0.5]],
                        aA = [1.0],
                        aB = [1.0])

rng = DiscreteDataGenerator(params)
@show data = generate(rng)


# params = DataParameters( nA = 1000, nB = 1000,
#                          mA = [0.0], mB = [0.0],
#                          covA = ones(1,1), covB = ones(1, 1))
#  
# rng = DiscreteDataGenerator(params, n = 1000, scenario = 1)
#  
# # data = generate(rng)



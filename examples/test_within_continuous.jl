using OptimalTransportDataIntegration

params = DataParameters(nA = 1000, nB = 1000)

rng = DataGenerator(params, n = 1000)

data = generate(rng)

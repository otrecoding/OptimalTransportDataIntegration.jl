using OptimalTransportDataIntegration

params = DataParameters(mA = [0.0], 
                        mB = [6.0],
                        covA = ones(1,1),
                        covB = ones(1,1),
                        aA = [1.0],
                        aB = [1.0],
                        r2 = 0.6)
 
rng = ContinuousDataGenerator(params; scenario = 2)
data = generate(rng)

@show accuracy(otrecod(data, SimpleLearning()))
@show accuracy(otrecod(data, JointOTWithinBase()))
@show accuracy(otrecod(data, JointOTBetweenBases()))
@show accuracy(otrecod(data, JointOTBetweenBasesWithoutYZ()))
@show accuracy(otrecod(data, JointOTBetweenBasesRefJDOT()))
@show accuracy(otrecod(data, JointOTBetweenBasesRefOTDAx()))
@show accuracy(otrecod(data, JointOTBetweenBasesRefOTDAyz()))
@show accuracy(otrecod(data, JointOTBetweenBasesRefOTDAyzPred()))
 
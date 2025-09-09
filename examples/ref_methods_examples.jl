using OptimalTransportDataIntegration

params = DataParameters(mA = [0.0], 
                        mB = [6.0],
                        covA = ones(1,1),
                        covB = ones(1,1),
                        aA = [1.0],
                        aB = [1.0],
                        r2 = 0.6)
 
rng = ContinuousDataGenerator(params; scenario = 1)
data = generate(rng)

sol = otrecod(data, SimpleLearning())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTWithinBase())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTBetweenBases())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTBetweenBasesWithoutYZ())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTBetweenBasesRefJDOT())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTBetweenBasesRefOTDAx())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTBetweenBasesRefOTDAyz())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTBetweenBasesRefOTDAyzPred())
@show accuracy(sol)
@show confusion_matrix(sol.yb_pred, sol.yb_true)
 
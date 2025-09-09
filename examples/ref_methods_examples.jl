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
accuracy(sol)
confusion_matrix(sol.yb_pred, sol.yb_true)
sol = otrecod(data, JointOTWithinBase())
accuracy(sol)
confusion_matrix(sol.yb_pred, sol.yb_true)
accuracy(sol)
sol = accuracy(otrecod(data, JointOTBetweenBases()))
accuracy(sol)
confusion_matrix(sol.yb_pred, sol.yb_true)
sol = accuracy(otrecod(data, JointOTBetweenBasesWithoutYZ()))
accuracy(sol)
confusion_matrix(sol.yb_pred, sol.yb_true)
sol = accuracy(otrecod(data, JointOTBetweenBasesRefJDOT()))
accuracy(sol)
confusion_matrix(sol.yb_pred, sol.yb_true)
sol = accuracy(otrecod(data, JointOTBetweenBasesRefOTDAx()))
confusion_matrix(sol.yb_pred, sol.yb_true)
sol = accuracy(otrecod(data, JointOTBetweenBasesRefOTDAyz()))
accuracy(sol)
confusion_matrix(sol.yb_pred, sol.yb_true)
sol = accuracy(otrecod(data, JointOTBetweenBasesRefOTDAyzPred()))
confusion_matrix(sol.yb_pred, sol.yb_true)
 
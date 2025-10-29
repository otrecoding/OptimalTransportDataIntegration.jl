using OptimalTransportDataIntegration
using Distributions
using DocStringExtensions
using Parameters
using Printf

#params = DataParameters(
# nA = 1000,
# nB = 1000,
# mA = [0.0],
# mB = [2.0],
# covA = ones(1,1),
# covB = ones(1,1),
# aA = [1.0],
# aB = [1.0],
# r2 = 0.9,
# pA = [[0.25, 0.25,0.25,0.25]],
# pB= [[0.7, 0.1,0.1,0.1]]  )
# ici JointOTBetweenBaseswithpred marche mieux
#params = DataParameters(
#    nA = 1000,
#    nB = 1000,
#    mA = [0.0,0.0,0.0],
#    mB = [0.0,0.0,0.0],
#    covA = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
#    covB = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
#    aA = [1.0,1.0,1.0],
#    aB = [1.0,1.0,1.0],
#    r2 = 0.9,
#    pA = [
#               [0.5, 0.5],                # 2 categories
#               [1/3,1/3,1/3],             # 3 categories
#               [0.25, 0.25, 0.25, 0.25],  # 4 categories
#          ],
#    pB= [
#               [0.1, 0.9],                # 2 categories
#               [0.1,0.1,0.8],             # 3 categories
#               [0.1, 0.1, 0.1, 0.7],  # 4 categories
#          ]  )

#ot within marche un peu mieux
params = DataParameters(
    nA = 1000,
    nB = 1000,
    mA = [0.0, 0.0, 0.0],
    mB = [0.0, 0.0, 0.0],
    covA = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
    covB = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
    aA = [1.0, 1.0, 1.0],
    aB = [1.0, 1.0, 1.0],
    r2 = 0.6,
    pA = [
        [0.5, 0.5],                # 2 categories
        [1 / 3, 1 / 3, 1 / 3],             # 3 categories
        [0.25, 0.25, 0.25, 0.25],  # 4 categories
    ],
    pB = [
        [0.8, 0.2],                # 2 categories
        [1 / 3, 1 / 3, 1 / 3],             # 3 categories
        [0.25, 0.25, 0.25, 0.25],  # 4 categories
    ]
)
############avec des variables binaire joint_ot_between_bases marche mieux que JointOTBetweenBaseswithpred (les X sont en one hot encoding)
# params = DataParameters(
#    nA = 1000,
#    nB = 1000,
#    mA = [0.0,0.0,0.0],
#    mB = [0.0,0.0,0.0],
#    covA = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
#    covB = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0],
#    aA = [1.0,1.0,1.0],
#    aB = [1.0,1.0,1.0],
#    r2 = 0.9,
#    pA = [
#               [0.5, 0.5],                # 2 categories
#               [0.5,0.5],             # 3 categories
#               [0.5,0.5],  # 4 categories
#          ],
#    pB= [
#               [0.8, 0.2],                # 2 categories
#               [0.8,0.2],             # 3 categories
#               [0.8,0.2],  # 4 categories
#          ]  )
rng = DiscreteDataGenerator(params; scenario = 1)
data = generate(rng)


@show accuracy(otrecod(data, SimpleLearning()))

@show accuracy(otrecod(data, JointOTWithinBase(lambda = 0.0, alpha = 0.0)))

@show accuracy(otrecod(data, JointOTBetweenBaseswithpred(reg = 0.0)))

@show accuracy(otrecod(data, JointOTBetweenBases(reg_m1 = 0)))

@show accuracy(otrecod(data, JointOTBetweenBasesc(reg_m1 = 0)))

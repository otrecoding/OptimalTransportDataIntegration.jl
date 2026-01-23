using Distances
using OptimalTransportDataIntegration

params = DataParameters()

data = generate_real_discrete_data(params)


database = data.database

Xnames = names(data, r"^X")

X = Matrix(data[!, Xnames])
Y = Vector(data.Y)
Z = Vector(data.Z)

Ylevels = 1:4
Zlevels = 1:3

distance = Euclidean()
instance = Instance(database, X, Y, Ylevels, Z, Zlevels, distance)

lambda = 0.1
alpha = 0.1
percent_closest = 0.2

sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

YB, ZA = compute_pred_error!(sol, instance, false)

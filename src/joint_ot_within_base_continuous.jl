function joint_ot_within_base_continuous(
        data;
        lambda = 0.392,
        alpha = 0.714,
        percent_closest = 0.2,
        distance = Euclidean(),
    )

    digitize(x, bins) = searchsortedlast.(Ref(bins), x)

    XA = subset(data, :database => x -> x .== 1.0)
    XB = subset(data, :database => x -> x .== 2.0)

    b1 = quantile(data.X1, collect(0.1:0.1:0.9))
    bins11 = vcat(-Inf, b1, +Inf)

    X11 = digitize(XA.X1, bins11)
    X21 = digitize(XB.X1, bins11)

    b1 = quantile(data.X2, collect(0.1:0.1:0.9))
    bins12 = vcat(-Inf, b1, +Inf)

    X12 = digitize(XA.X2, bins12)
    X22 = digitize(XB.X2, bins12)

    b1 = quantile(data.X3, collect(0.1:0.1:0.9))
    bins13 = vcat(-Inf, b1, +Inf)

    X13 = digitize(XA.X3, bins13)
    X23 = digitize(XB.X3, bins13)

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)

    X = hcat(X1, X2, X3)
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    Ylevels = 1:4
    Zlevels = 1:3

    database = data.database

    instance = Instance(database, X, Y, Ylevels, Z, Zlevels, distance)

    sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

    return compute_pred_error!(sol, instance, false)

end

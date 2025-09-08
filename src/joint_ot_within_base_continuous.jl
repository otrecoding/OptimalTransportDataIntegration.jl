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

    for col in names(data, r"^X")

        b = quantile(data[!, col], collect(0.1:0.1:0.9))
        bins = vcat(-Inf, b, +Inf)

        X1 = digitize(XA.X1, bins)
        X2 = digitize(XB.X1, bins)

        if col == "X1"
           X = vcat(X1, X2)
        else
           X = hcat(X, vcat(X1, X2))
        end

    end

    Y = Vector(data.Y)
    Z = Vector(data.Z)

    Ylevels = 1:4
    Zlevels = 1:3

    database = data.database

    instance = Instance(database, X, Y, Ylevels, Z, Zlevels, distance)

    sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

    return compute_pred_error!(sol, instance, false)

end

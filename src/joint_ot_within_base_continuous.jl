function joint_ot_within_base_continuous(
        data;
        lambda = 0.392,
        alpha = 0.714,
        percent_closest = 0.2,
        distance = Euclidean(),
        Ylevels = 1:4,
        Zlevels = 1:3
    )

    digitize(x, bins) = searchsortedlast.(Ref(bins), x)

    XA = subset(data, :database => x -> x .== 1)
    XB = subset(data, :database => x -> x .== 2)

    X = Vector{Float64}[]
    for col in names(data, r"^X")

        b = quantile(data[!, col], collect(0.25:0.25:0.75))
        bins = vcat(-Inf, b, +Inf)

        X1 = digitize(XA[!, col], bins)
        X2 = digitize(XB[!, col], bins)

        push!(X, vcat(X1, X2))

    end

    Y = Vector(data.Y)
    Z = Vector(data.Z)

    database = data.database

    instance = Instance(database, stack(X), Y, Ylevels, Z, Zlevels, distance)

    sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

    return compute_pred_error!(sol, instance, false)

end

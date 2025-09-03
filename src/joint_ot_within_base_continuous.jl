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

    cols = names(data, r"^X")   # toutes les colonnes X1, X2, ..., Xp
    p = length(cols)

    Xlist = []  # va contenir les colonnes discrétisées

    for col in cols
    # bornes de discrétisation basées sur les quantiles de la colonne dans data
        b = quantile(data[!, col], collect(0.1:0.1:0.9))
        bins = vcat(-Inf, b, +Inf)

    # discrétisation pour XA et XB
        XA_d = digitize(XA[!, col], bins)
        XB_d = digitize(XB[!, col], bins)

    # concaténer les deux
        push!(Xlist, vcat(XA_d, XB_d))
end

# Construire la matrice finale
    X = hcat(Xlist)  
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    Ylevels = 1:4
    Zlevels = 1:3

    database = data.database

    instance = Instance(database, X, Y, Ylevels, Z, Zlevels, distance)

    sol = OptimalTransportDataIntegration.ot_joint(instance, alpha, lambda, percent_closest)

    return compute_pred_error!(sol, instance, false)

end

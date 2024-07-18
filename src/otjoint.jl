function otjoint( data; lambda_reg = 0.392, maxrelax = 0.714, percent_closest = 0.2)

    database = data.database

    Xnames = ["X1", "X2", "X3"]

    X = one_hot_encoder(Matrix(data[!, Xnames]))
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    instance = Instance( database, X, Y, Z, Hamming())

    sol = ot_joint(instance, maxrelax, lambda_reg, percent_closest)
    compute_pred_error!(sol, instance, false)

    return 1 - sol.errorpredavg

end

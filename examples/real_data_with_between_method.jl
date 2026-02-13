using FreqTables
using MultivariateStats
using OptimalTransportDataIntegration
using StatsBase

include(joinpath(@__DIR__, "read_real_data.jl"))
include(joinpath(@__DIR__, "discretize.jl"))

function main_between()

    data = read_data()
    X, Xmdn = discretize(data)
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    cols = names(dba, r"^X")

    X_hot = one_hot_encoder(X)
    Y = Vector(data.Y)
    Z = Vector(data.Z)

    continuous_cols = [:X1, :X4]
    categorical_cols = [:X2, :X3, :X5]

    for name in  categorical_cols
        lev = union(unique(dba[!, name]), unique(dbb[!, name]))
        dba[!, name] = categorical(dba[!, name], levels = lev)
        dbb[!, name] = categorical(dbb[!, name], levels = lev)
    end

    for name in continuous_cols 
        dba[!, name] = Float32.(dba[!, name])
        dbb[!, name] = Float32.(dbb[!, name])
    end


    X_hot = Matrix{Float32}(X_hot)
    Xdf = DataFrame(X_hot, Symbol.("X" .* string.(1:size(X_hot, 2))))

    data2 = hcat(Xdf, data[:, [:Y, :Z, :database]])

    dba = subset(data2, :database => ByRow(==(1)))
    dbb = subset(data2, :database => ByRow(==(2)))

    result = otrecod(data2, JointOTBetweenBases(reg = 0.001, reg_m1 = 0.01, reg_m2 = 0.01, Ylevels = 1:4, Zlevels = 1:6))

    tab1 = FreqTables.freqtable(dba.Y, result.za_pred)
    tab2 = FreqTables.freqtable(dbb.Z, result.yb_pred)

    return tab1, tab2 

end

@time main_between()

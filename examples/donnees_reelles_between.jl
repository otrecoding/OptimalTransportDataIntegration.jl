using OptimalTransportDataIntegration

include(joinpath(@__DIR__, "read_data.jl"))
include(joinpath(@__DIR__, "discretize.jl"))
include(joinpath(@__DIR__, "gower_distance.jl"))

function main_between()

    data = read_data()
    X, Xmdn = discretize(data)
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    cols = names(dba, r"^X")   
    
    X_hot = one_hot_encoder(X)
    Y = Vector(data.Y)
    Z = Vector(data.Z)
    base = data.database
    
    nA = size(dba, 1)
    nB = size(dbb, 1)
    
    for name in ["X2","X3","X5"]
        lev = union(unique(dba[!, name]), unique(dbb[!, name]))
        dba[!, name] = categorical(dba[!, name], levels=lev)
        dbb[!, name] = categorical(dbb[!, name], levels=lev)
    end             
    
    for name in ["X1","X4"]
        dba[!, name] = Float64.(dba[!, name])
        dbb[!, name] = Float64.(dbb[!, name])
    end

    A = dba[:, cols]
    B = dbb[:, cols]
    # --- Exemple : DataFrames A et B déjà définis ---
    df_all = vcat(A, B)
    
    # --- Colonnes continues et catégorielles ---
    continuous_cols = [:X1, :X4]
    categorical_cols = [:X2, :X3, :X5]
    
    # 1. Standardisation des variables continues
    df_cont_std = DataFrame()
    for c in continuous_cols
        x = skipmissing(df_all[!, c])
        μ = mean(x)
        σ = std(x)
        df_cont_std[!, c] = (df_all[!, c] .- μ) ./ σ
    end
    
    # 2. One-hot encoding des variables catégorielles
    df_dummy = DataFrame()
    for c in categorical_cols
        df_cat = categorical(df_all[!, c])
        for lvl in levels(df_cat)
            col_name = Symbol(string(c) * "_" * string(lvl))
            df_dummy[!, col_name] = df_cat .== lvl
        end
    end
    
    # 3. Pondération FAMD
    df_dummy_famd = DataFrame()
    for c in names(df_dummy)
        p = mean(df_dummy[!, c])
        df_dummy_famd[!, c] = df_dummy[!, c] ./ sqrt(p)
    end
    
    # 4. Construction de la matrice finale
    Xn = hcat(Matrix(df_cont_std), Matrix(df_dummy_famd))
    
    # 5. ACP (PCA)
    ncomp = 13
    model = fit(PCA, Xn'; maxoutdim=ncomp)
    
    # 6. Coordonnées factorielles
    coords_all = MultivariateStats.transform(model, Xn')  #
    # --- coords_all contient les coordonnées FAMD de df_all ---
    coords_all2 = coords_all'
    
    # %%
    T = Float64
    X_hot=Matrix{T}(X_hot)
    Xdf = DataFrame(X_hot, Symbol.("X" .* string.(1:size(X_hot, 2))))
    data2 = hcat(Xdf,data[:, [:Y, :Z, :database]])
    dba = subset(data2, :database => ByRow(==(1)))
    dbb = subset(data2, :database => ByRow(==(2)))
    A = otrecod(data2, JointOTBetweenBases(reg = 0.001, reg_m1= 0.01, reg_m2 = 0.01, Ylevels = 1:4, Zlevels = 1:6))
    @show tab1 = FreqTables.freqtable(dba.Y,A.za_pred)
    @show tab2 = FreqTables.freqtable(dbb.Z,A.yb_pred)
    
    T = Float64
    Xdf = DataFrame(coords_all2, Symbol.("X" .* string.(1:size(coords_all2, 2))))
    data3 = hcat(Xdf,data[:, [:Y, :Z, :database]])
    dba = subset(data3, :database => ByRow(==(1)))
    dbb = subset(data3, :database => ByRow(==(2)))
    A = otrecod(data3, JointOTBetweenBasesWithPredictors(reg = 0.001, reg_m1= 0.01, reg_m2 = 0.01, Ylevels = 1:4, Zlevels = 1:6))
    @show tab1 = FreqTables.freqtable(dba.Y,A.za_pred)
    @show tab2 = FreqTables.freqtable(dbb.Z,A.yb_pred)
    
end

@time main_between()

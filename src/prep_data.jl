export prep_data

function prep_data( df :: DataFrame )

    data = one_hot_encoder(df[!, [:X1, :X2, :X3]])
    data.database = df.database
    data.Y = df.Y
    data.Z = df.Z
    
    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    
    YBtrue = dbb.Y
    ZAtrue = dba.Z
    
    X = Matrix{Int}(data)
    Y = Vector{Int}(df.Y)
    Z = Vector{Int}(df.Z)
    
    Xnames = [:X1_1,:X2_1,:X2_2, :X3_1, :X3_2, :X3_3]

    XA = Matrix{Int}(dba[!, Xnames])
    XB = Matrix{Int}(dbb[!, Xnames])

    YA = one_hot_encoder(dba.Y)
    ZA = one_hot_encoder(dba.Z)
    YB = one_hot_encoder(dbb.Y)
    ZB = one_hot_encoder(dbb.Z)
    
    return Xnames, X, Y, Z, XA, YA, XB, ZB, YBtrue, ZAtrue

end



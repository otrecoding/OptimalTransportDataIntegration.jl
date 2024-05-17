using Distributions
using DataFrames

export generate_xcat_ycat

"""
$(SIGNATURES)

Function to generate data where X and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

"""
function  generate_xcat_ycat( params )

    d = MvNormal(params.mA, params.covA)    
    X = rand(d, params.nA)
    
    px1cc = cumsum(params.px1c)[1:end-1]
    px2cc = cumsum(params.px2c)[1:end-1]
    px3cc = cumsum(params.px3c)[1:end-1]
    
    qx1c = quantile(Normal(0.0, 1.0), px1cc)
    qx2c = quantile(Normal(0.0, 1.0), px2cc)
    qx3c = quantile(Normal(0.0, 1.0), px3cc)
    
    bins11 = vcat(minimum(X[1,:]) - 100, qx1c, maximum(X[1,:]) + 100)
    bins12 = vcat(minimum(X[2,:]) - 100, qx2c, maximum(X[2,:]) + 100)
    bins13 = vcat(minimum(X[3,:]) - 100, qx3c, maximum(X[3,:]) + 100)
    
    X = rand(d, params.nA)
    
    X1 = digitize(X[1,:], bins11)
    X2 = digitize(X[2,:], bins12)
    X3 = digitize(X[3,:], bins13)
    
    X1c = to_categorical(X1)[2:end,:]
    X2c = to_categorical(X2)[2:end,:]
    X3c = to_categorical(X3)[2:end,:]
    
    Y1 = vcat(X1c, X2c, X3c)' * params.aA
    
    b11 = quantile(Y1, [0.25, 0.5, 0.75])
    b22 = quantile(Y1, [1 / 3, 2 / 3])
    binsY11 = vcat(minimum(Y1) - 100, b11, maximum(Y1) + 100)
    binsY22 = vcat(minimum(Y1) - 100, b22, maximum(Y1) + 100)
    
    XA = rand(d, params.nA)
    XB = rand(d, params.nB)
    
    X11 = digitize(XA[1,:], bins11)
    X21 = digitize(XB[1,:], bins11)
    X12 = digitize(XA[2,:], bins12)
    X22 = digitize(XB[2,:], bins12)
    X13 = digitize(XA[3,:], bins13)
    X23 = digitize(XB[3,:], bins13)
    
    X11c = to_categorical(X11)[2:end,:]
    X21c = to_categorical(X21)[2:end,:]
    X12c = to_categorical(X12)[2:end,:]
    X22c = to_categorical(X22)[2:end,:]
    X13c = to_categorical(X13)[2:end,:]
    X23c = to_categorical(X23)[2:end,:]
    
    Y1 = vcat(X11c, X12c, X13c)' * params.aA
    Y2 = vcat(X21c, X22c, X23c)' * params.aB
    
    Y11 = digitize(Y1, binsY11)
    Y12 = digitize(Y1, binsY22)
    binsY11eps = [x + params.eps for x in binsY11]
    Y21 = digitize(Y2, binsY11eps)
    binsY22eps = [x + params.eps for x in binsY22]
    Y22 = digitize(Y2, binsY22eps)
    
    YY = vcat(Y11, Y21)
    ZZ = vcat(Y12, Y22)
    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)
    
    p = params.p
    
    py = [1 - p,(p - p^2) / 2,(p - p^2) / 2,
        p^2 / 4,p^2 / 4,p^2 / 4,p^2 / 4,]
    pz = [1 - p, (p - p^2) / 2, (p - p^2) / 2, p^2 / 3, p^2 / 3, p^2 / 3]
    
    U = rand(Multinomial(1, py), params.nA + params.nB)
    V = rand(Multinomial(1, pz), params.nA + params.nB)
    
    UU = U' * Vector(0:6)
    VV = V' * Vector(0:5)
    
    Y = Vector(0:(params.nA + params.nB - 1))
    Z = Vector(0:(params.nA + params.nB - 1))
    
    Y[UU .== 0] .= YY[UU .== 0]
    Y[UU .== 1] .= YY[UU .== 1] .- 1
    Y[UU .== 2] .= YY[UU .== 2] .+ 1
    Y[UU .== 3] .= 1
    Y[UU .== 4] .= 2
    Y[UU .== 5] .= 3
    Y[UU .== 6] .= 4
    Y[Y .> 4] .= 4
    Y[Y .< 1] .= 1
    Z[VV .== 0] .= ZZ[VV .== 0]
    Z[VV .== 1] .= ZZ[VV .== 1] .- 1
    Z[VV .== 2] .= ZZ[VV .== 2] .+ 1
    Z[VV .== 3] .= 1
    Z[VV .== 4] .= 2
    Z[VV .== 5] .= 3
    Z[Z .> 3] .= 3
    Z[Z .< 1] .= 1
    
    data = DataFrame(hcat(X1, X2, X3) .- 1, [:X1, :X2, :X3])
    data.Y = Y
    data.Z = Z
    data.database = vcat(fill(1,params.nA), fill(2, params.nB))
    data

end




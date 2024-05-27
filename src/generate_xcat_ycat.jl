using Distributions
using DataFrames

export generate_xcat_ycat

"""
$(SIGNATURES)

Function to generate data where X and (Y,Z) are categoricals

the function return a Dataframe with X1, X2, X3, Y, Z and the database id.

"""
function generate_xcat_ycat( params )

    dA = MvNormal(params.mA, params.covA)    
    dB = MvNormal(params.mB, params.covB)    

    X_glob1 = rand(dA, params.nA)
    X_glob2 = rand(dB, params.nB)
    
    @show px1cc = cumsum(params.px1c)[1:end-1]
    @show px2cc = cumsum(params.px2c)[1:end-1]
    @show px3cc = cumsum(params.px3c)[1:end-1]
    
    @show qx1c = quantile.(Normal(0.0, 1.0), px1cc)
    @show qx2c = quantile.(Normal(0.0, 1.0), px2cc)
    @show qx3c = quantile.(Normal(0.0, 1.0), px3cc)
    
    bins11 = vcat(minimum(X_glob1[1,:]) - 100, qx1c, maximum(X_glob1[1,:]) + 100)
    bins12 = vcat(minimum(X_glob1[2,:]) - 100, qx2c, maximum(X_glob1[2,:]) + 100)
    bins13 = vcat(minimum(X_glob1[3,:]) - 100, qx3c, maximum(X_glob1[3,:]) + 100)
    
    X11 = digitize(X_glob1[1, :], bins11)
    X12 = digitize(X_glob1[2, :], bins12)
    X13 = digitize(X_glob1[3, :], bins13)

    X21 = digitize(X_glob2[1, :], bins11)
    X22 = digitize(X_glob2[2, :], bins12)
    X23 = digitize(X_glob2[3, :], bins13)

    X11c = to_categorical(X11)[2:end,:]
    X21c = to_categorical(X21)[2:end,:]
    X12c = to_categorical(X12)[2:end,:]
    X22c = to_categorical(X22)[2:end,:]
    X13c = to_categorical(X13)[2:end,:]
    X23c = to_categorical(X23)[2:end,:]

    X1 = vcat(X11, X21)
    X2 = vcat(X12, X22)
    X3 = vcat(X13, X23)

    Y1 = vcat(X11c, X12c, X13c)' * params.aA
    Y2 = vcat(X21c, X22c, X23c)' * params.aB

    p = params.p

    py = [
        1 - p,
        (p - p^2) / 2,
        (p - p^2) / 2,
        (p^2 - p^3) / 2,
        (p^2 - p^3) / 2,
        p^3 / 4,
        p^3 / 4,
        p^3 / 4,
        p^3 / 4
    ]

    U = rand(Multinomial(1, py), params.nA)
    V = rand(Multinomial(1, py), params.nB)

    UU = vec(sum(U .* collect(0:8), dims=1))
    VV = vec(sum(V .* collect(0:8), dims=1))

    Y = collect(0:(params.nA-1))
    Z = collect(0:(params.nB-1))

    Y[UU .== 0] .= Y1[UU .== 0]
    Y[UU .== 1] .= Y1[UU .== 1] .- 1
    Y[UU .== 2] .= Y1[UU .== 2] .+ 1
    Y[UU .== 3] .= Y1[UU .== 3] .+ 2
    Y[UU .== 4] .= Y1[UU .== 4] .- 2
    Y[UU .== 5] .= 0
    Y[UU .== 6] .= 1
    Y[UU .== 7] .= 2
    Y[UU .== 8] .= 3
    Y[Y .> 3] .= 3
    Y[Y .< 0] .= 0
    
    Z[VV .== 0] .= Y2[VV .== 0]
    Z[VV .== 1] .= Y2[VV .== 1] .- 1
    Z[VV .== 2] .= Y2[VV .== 2] .+ 1
    Z[VV .== 3] .= Y2[VV .== 3] .+ 2
    Z[VV .== 4] .= Y2[VV .== 4] .- 2
    Z[VV .== 5] .= 0
    Z[VV .== 6] .= 1
    Z[VV .== 7] .= 2
    Z[VV .== 8] .= 3
    Z[Z .> 3] .= 3
    Z[Z .< 0] .= 0

    b11 = quantile(Y, [0.25, 0.5, 0.75])
    b22 = quantile(Z, [1 / 3, 2 / 3])

    binsY11 = vcat(minimum(Y) - 100, b11, maximum(Y) + 100)
    binsY22 = vcat(minimum(Z) - 100, b22, maximum(Z) + 100)
    
    Y11 = digitize(Y1, binsY11)
    Y12 = digitize(Y1, binsY22)
    binsY11eps = binsY11 .+ params.eps
    Y21 = digitize(Y2, binsY11eps)
    binsY22eps = binsY22 .+ params.eps
    Y22 = digitize(Y2, binsY22eps)
    
    data = DataFrame(hcat(X1, X2, X3) .- 1, [:X1, :X2, :X3])
    data.Y = vcat(Y11, Y21)
    data.Z = vcat(Y12, Y22)
    data.database = vcat(fill(1,params.nA), fill(2, params.nB))
    data

end




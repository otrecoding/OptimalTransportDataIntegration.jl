using CategoricalArrays
using DataFrames
import Flux
import Statistics: median

function discretize( data )

    digitize(x, bins) = searchsortedlast.(Ref(bins), x)

    dba = subset(data, :database => ByRow(==(1)))
    dbb = subset(data, :database => ByRow(==(2)))
    cols = names(dba, r"^X")   
    Ylevels = 1:4
    Zlevels = 1:6
    YA = Flux.onehotbatch(dba.Y, Ylevels)
    ZB = Flux.onehotbatch(dbb.Z, Zlevels)
    
    XA = dba[:, cols]
    XB = dbb[:, cols]
    
    X = Vector{Int}[]
    Xmdn = Vector{Float64}[]
    cols = ["X1", "X4"]
    for col in cols
    
            b = quantile(data[!, col], collect(0.25:0.25:0.75))
            bins = vcat(-Inf, b, +Inf)
    
            X1 = digitize(XA[!, col], bins)
            X2 = digitize(XB[!, col], bins)
            push!(X, vcat(X1, X2))
    
            X1mdn = zeros(Float32, size(X1, 1))
            for i in unique(X1)
                mdn = median(XA[X1 .== i, col])
                X1mdn[X1 .== i] .= mdn
            end
    
            X2mdn = zeros(Float32, size(X2, 1))
            for i in unique(X2)
                mdn = median(XB[X2 .== i, col])
                X2mdn[X2 .== i] .= mdn
            end
    
            push!(Xmdn, vcat(X1mdn, X2mdn))
    
    end
    
    cols = ["X2", "X3","X5"]
    for col in cols
    
            X1 = XA[!, col]
            X2 = XB[!, col]
            push!(X, vcat(X1, X2))
    
            X1mdn = XA[!, col]
    
            X2mdn = XB[!, col]
    
            push!(Xmdn, vcat(X1mdn, X2mdn))
    
    end
    X = stack(X)
    Xmdn = stack(Xmdn)

    return X, Xmdn

end

import Distances: PreMetric

export Gower

struct Gower <: PreMetric
    continuous::Vector{Symbol}
    categorical::Vector{Symbol}
    ranges::Dict{Symbol, Float64}   # range par variable continue
end

function Gower(continuous, categorical, df::DataFrame)
    ranges = Dict{Symbol, Float64}()
    for col in continuous
        coldata = skipmissing(df[!, col])
        ranges[col] = maximum(coldata) - minimum(coldata)
    end
    return Gower(continuous, categorical, ranges)
end

function Distances.evaluate(d::Gower, a::DataFrameRow, b::DataFrameRow)
    s = 0.0
    p = length(d.continuous) + length(d.categorical)

    # continues
    for col in d.continuous
        range = d.ranges[col]

        if range == 0 || ismissing(a[col]) || ismissing(b[col])
            contrib = 0.0
        else
            contrib = abs(a[col] - b[col]) / range
        end

        s += contrib
    end

    # catégories
    for col in d.categorical
        if ismissing(a[col]) || ismissing(b[col])
            s += 0.0
        else
            s += (a[col] == b[col] ? 0.0 : 1.0)
        end
    end

    return s / p
end

export pairwise_gower

function pairwise_gower(d::Gower, A::DataFrame, B::DataFrame)

    nA = nrow(A)
    nB = nrow(B)
    D = Matrix{Float64}(undef, nA, nB)

    for i in 1:nA
        row_i = A[i, :]
        for j in 1:nB
            row_j = B[j, :]
            D[i, j] = Distances.evaluate(d, row_i, row_j)
        end
    end

    return D
end

function digitize_int(XA, XB, column::Symbol)
    b = quantile(data[!, col], collect(0.25:0.25:0.75))
    bins = vcat(-Inf, b, +Inf)
    X1 = digitize(XA[!, col], bins)
    X2 = digitize(XB[!, col], bins)
    return vact(X1, X2)
end

function digitize_median(XA, XB, column::Symbol)

    b = quantile(data[!, col], collect(0.25:0.25:0.75))
    bins = vcat(-Inf, b, +Inf)
    X1 = digitize(XA[!, col], bins)
    X2 = digitize(XB[!, col], bins)

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

    return vcat(X1mdn, X2mdn)

end

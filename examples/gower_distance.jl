using Distances

struct GowerDF2 <: PreMetric
    continuous::Vector{Symbol}
    categorical::Vector{Symbol}
    ranges::Dict{Symbol,Float64}   # range par variable continue
end

function GowerDF2(continuous, categorical, df::DataFrame)
    ranges = Dict{Symbol,Float64}()
    for col in continuous
        coldata = skipmissing(df[!, col])
        ranges[col] = maximum(coldata) - minimum(coldata)
     end
    return GowerDF2(continuous, categorical, ranges)
end

function Distances.evaluate(d::GowerDF2, a::DataFrameRow, b::DataFrameRow)
    s = 0.0
    p = length(d.continuous) + length(d.categorical)

    # continues (version correcte Gower)
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

function pairwise_gower(d::GowerDF2, A::DataFrame, B::DataFrame)
    nA = nrow(A)
    nB = nrow(B)
    D = Matrix{Float64}(undef, nA, nB)

    for i in 1:nA
        row_i = A[i, :]
        for j in 1:nB
            row_j = B[j, :]
            D[i,j] = Distances.evaluate(d, row_i, row_j)
        end
    end

    return D
end


export one_hot_encoder

function one_hot_encoder(df::DataFrame)
    outnames = Symbol[]
    res = copy(df)
    for col in names(df)
        cates = sort(filter(x -> x != 0, unique(df[!, col])))
        outname = Symbol.(col, "_", cates)
        push!(outnames, outname...)
        transform!(res, @. col => ByRow(isequal(cates)) => outname)
    end
    return res[!, outnames]
end

function one_hot_encoder(x::AbstractMatrix)
    res = Vector{Int}[]
    for col in eachcol(x)
        levels = filter(x -> x != 0, sort(unique(col)))
        for level in levels
            push!(res, col .== level)
        end
    end
    return stack(res)
end

function one_hot_encoder(x::AbstractVector)::Matrix{Bool}

    (sort(filter(i -> i != 0, unique(x))) .== permutedims(x))'

end

function one_hot_encoder(x::AbstractVector, levels::AbstractVector)::Matrix{Bool}

    (sort(filter(i -> i != 0, levels)) .== permutedims(x))'

end

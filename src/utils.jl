export one_hot_encoder

function one_hot_encoder(df :: DataFrame)
    outnames = Symbol[]
    res = copy(df)
    for col in names(df)
        cates = sort(unique(df[!, col]))
        outname = Symbol.(col,"_", cates)
        push!(outnames, outname...)
        transform!(res, @. col => ByRow(isequal(cates)) => outname)
    end
    return res[!, outnames]
end

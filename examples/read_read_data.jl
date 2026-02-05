using DataFrames
using XLSX

function read_data()
    tbl = XLSX.readtable(joinpath(@__DIR__, "data.xlsx"), 1)  
    data = DataFrame(tbl)
    eltype.(eachcol(data))
    for c in names(data)
        data[!, c] = [
            x isa Number ? float(x) :            
            x isa Missing ? missing :       
            tryparse(Float64, String(x))
            for x in data[!, c]
        ]
    end
    for c in [:Y, :Z]
        data[!, c] = coalesce.(data[!, c], -1)
    end

    for c in [:Y, :Z, :database]
        data[!, c] = convert.(Union{Int64,Missing}, data[!, c])
    end
    return data
end

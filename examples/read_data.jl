using DataFrames
using XLSX

function read_data()
    tbl = XLSX.readtable(joinpath(@__DIR__, "data.xlsx"), 1)   # 1 = première feuille, ou utiliser le nom "Sheet1"
    data = DataFrame(tbl)
    eltype.(eachcol(data))
    for c in names(data)
        data[!, c] = [ 
            x isa Number ? float(x) :                 # si déjà numérique → convertir en Float64
            x isa Missing ? missing :                 # missing → garder
            tryparse(Float64, String(x))              # si string → tenter de parser
            for x in data[!, c]
        ]
    end
    for c in [:Y, :Z]
        data[!, c] = coalesce.(data[!, c], -1)
    end
    
    for c in [:Y, :Z, :database]
        data[!, c] = convert.(Union{Int64}, data[!, c])
    end
    return data
end


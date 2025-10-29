using OptimalTransportDataIntegration
import Flux
import Flux: Chain, Dense
#using DataFrames
function onehot(x::AbstractMatrix)
    res = Vector{Float32}[]
    for col in eachcol(x)
        levels = 1:4 #filter(x -> x != 0, sort(unique(col)))
        for level in levels
            push!(res, col .== level)
        end
    end
    return stack(res, dims = 1)
end

params = DataParameters(
    mA = [0.0],
    mB = [20.0],
    covA = ones(1, 1),
    covB = ones(1, 1),
    aA = [1.0],
    aB = [1.0]
)
data = generate(rng)
data
# dba = subset(data, :database => ByRow(==(1)))
# dbb = subset(data, :database => ByRow(==(2)))

# colnames = names(data, r"^X")
# XA = onehot(Matrix(dba[!, colnames]))
# XB = onehot(Matrix(dbb[!, colnames]))

YA = Flux.onehotbatch(dba.Y, Ylevels)
ZB = Flux.onehotbatch(dbb.Z, Zlevels)

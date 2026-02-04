using CategoricalArrays
using OptimalTransportDataIntegration

params = DataParameters()
data = generate_mixed_data(params)

data.X1 = convert(Vector{Int}, data.X1)
data.X2 = convert(Vector{Int}, data.X2)
data.X3 = convert(Vector{Int}, data.X3)
data.X4 = convert(Vector{Float32}, data.X4)
data.X5 = convert(Vector{Float32}, data.X5)
data.X6 = convert(Vector{Float32}, data.X6)
data.Y = convert(Vector{Union{Missing, Int}}, data.Y)
data.Z = convert(Vector{Union{Missing, Int}}, data.Z)

show(data)

xcols = names(data, r"^X")
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))
XA = dba[:, xcols]
XB = dbb[:, xcols]

cols = names(data, r"^X")
for name in ["X1", "X2", "X3"]
    lev = union(unique(Int.(dba[!, name])), unique(Int.(dbb[!, name])))
    dba[!, name] = categorical(dba[!, name], levels = lev)
    dbb[!, name] = categorical(dbb[!, name], levels = lev)
end


A = dba[:, cols]
B = dbb[:, cols]
C = vcat(A, B)
dist = Gower([:X4, :X5, :X6], [:X1, :X2, :X3], C)


D = pairwise_gower(dist, A, B)

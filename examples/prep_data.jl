# +
using OptimalTransportDataIntegration
using DataFrames
params = DataParameters(nA = 1000, nB = 500)

df = generate_xcat_ycat(params)

outnames = Symbol[]
res = copy(df)
for col in ["X1", "X2", "X3"]
    cates = sort(unique(df[!, col]))
    outname = Symbol.(col,"_", cates)
    push!(outnames, outname...)
    transform!(res, @. col => ByRow(isequal(cates)) => outname)
end
data = res[!, outnames]
data.database = df.database
data.Y = df.Y
data.Z = df.Z
dba = subset(data, :database => ByRow(==(1)))
dbb = subset(data, :database => ByRow(==(2)))

YBtrue = dbb.Y
ZAtrue = dba.Z

X = Matrix{Int}(data)
Y = Vector{Int}(df.Y)
Z = Vector{Int}(df.Z)
# -

Xnames = [:X1_1,:X2_1,:X2_2, :X3_1, :X3_2, :X3_3]
XA = dba[!, Xnames]
XB = dbb[!, Xnames]

YA = one_hot_encoder(dba.Y)
ZA = one_hot_encoder(dba.Z)
ZB = one_hot_encoder(dbb.Z)
YB = one_hot_encoder(dbb.Y)

end

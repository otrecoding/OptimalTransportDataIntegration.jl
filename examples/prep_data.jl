# +
using OptimalTransportDataIntegration
using DataFrames
params = DataParameters(nA = 1000, nB = 500)

df = generate_xcat_ycat(params)

# -

outnames = Symbol[]
res = copy(df)
for col in ["X1", "X2", "X3"]
    cates = sort(unique(df[!, col]))
    outname = Symbol.(col,"_", cates)
    push!(outnames, outname...)
    transform!(res, @. col => ByRow(isequal(cates)) => outname)
end
data = res[!, outnames]

dba = data[!, data.database]

# +
dbb = data[!, data.database == 2]

YBtrue = dbb.Y.values
ZAtrue = dba.Z.values

Xnames = ["X1_1", "X2_1", "X2_2", "X3_1", "X3_2", "X3_3"]

X = Matrix{Int}(data[!, Xnames])
Y = Vector{Int}(data.Y)
Z = Vector{Int}(data.Z)

XA = dba[!, Xnames]
XB = dbb[!, Xnames]

YA = pd.get_dummies(dba.Y, dtype=np.int32).values
ZB = pd.get_dummies(dbb.Z, dtype=np.int32).values
ZA = pd.get_dummies(dba.Z, dtype=np.int32).values
YB = pd.get_dummies(dbb.Y, dtype=np.int32).values

return Xnames, X, Y, Z, XA, YA, XB, ZB, YBtrue, ZAtrue
# -

end

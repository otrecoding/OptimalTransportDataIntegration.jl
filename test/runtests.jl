using OptimalTransportDataIntegration
using Test

params = DataParameters(nA = 1000, nB = 500)

data = generate_xcat_ycat(params)

@test sort(unique(data.Y)) ≈ [1, 2, 3, 4]
@test sort(unique(data.Z)) ≈ [1, 2, 3]


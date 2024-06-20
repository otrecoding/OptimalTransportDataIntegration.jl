using OptimalTransportDataIntegration
import OptimalTransportDataIntegration: unbalanced_modality, otjoint, simple_learning
using Test

params = DataParameters(nA = 1000, nB = 500)

data = generate_xcat_ycat(params)

@test sort(unique(data.Y)) ≈ [1, 2, 3, 4]
@test sort(unique(data.Z)) ≈ [1, 2, 3]

@test unbalanced_modality(data) > 0.8
@test otjoint( data; lambda_reg = 0.392, maxrelax = 0.714, percent_closest = 0.2) > 0.8
@test simple_learning( data; hidden_layer_size = 10,  learning_rate = 0.01, batchsize=64, epochs = 500) > 0.8

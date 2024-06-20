using  OptimalTransportDataIntegration
import OptimalTransportDataIntegration: unbalanced_modality, otjoint, simple_learning

using ProgressMeter

function run_simulations( simulations )

    params = DataParameters(nA=1000, nB=1000, mB=[2,0,0], eps=0, p=0.2)
    prediction_quality = []
    simulations = 100
    @showprogress 1 for i in 1:simulations
       data = generate_xcat_ycat(params)
      
       err1 = unbalanced_modality(data)
       err2 = otjoint( data; lambda_reg = 0.392, maxrelax = 0.714, percent_closest = 0.2)
       err3 = simple_learning( data; hidden_layer_size = 10,  learning_rate = 0.01, batchsize=512, epochs = 500)

       push!(prediction_quality, (err1, err2, err3))

    end

    prediction_quality

end

run_simulations( 100 )

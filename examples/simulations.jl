using  OptimalTransportDataIntegration
using DataFrames

# +
using ProgressMeter

function run_simulations( simulations )

    params = DataParameters(nA=1000, nB=1000, mB=[1,0,0], eps=0, p=0.3)
    prediction_quality = []
    
    @showprogress 1 for i in 1:simulations
        
       data = generate_xcat_ycat(params)
      
       err1 = OptimalTransportDataIntegration.unbalanced_modality(data)
       err2 = OptimalTransportDataIntegration.otjoint( data; lambda_reg = 0.392, maxrelax = 0.714, percent_closest = 0.2)
       err3 = OptimalTransportDataIntegration.simple_learning( data; hidden_layer_size = 10,  learning_rate = 0.01, batchsize=64, epochs = 500)

       push!(prediction_quality, (err1, err2, err3))

    end

    prediction_quality

end
# -

run_simulations( 10 )



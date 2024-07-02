using  OptimalTransportDataIntegration
using DataFrames

using JSON, CSV

# +

data = CSV.read("tab_otjoint_00_00.csv", DataFrame) # generated with Python code
@time OptimalTransportDataIntegration.unbalanced_modality(data)
# -

function read_params( jsonfile )
    
    data = JSON.parsefile(jsonfile)
    
    params = Dict()
    
    params[:nA] = Int(data["nA"])
    params[:nB] = Int(data["nB"])
    
    params[:aA] = Float64.(data["aA"])
    params[:aB] = Float64.(data["aB"])
    
    params[:mA] = vec(Float64.(data["mA"]))
    params[:mB] = vec(Float64.(data["mB"]))
    
    params[:covA] = stack([Float64.(x) for x in data["covA"]])
    params[:covB] = stack([Float64.(x) for x in data["covB"]])
  
    params[:px1c] = Float64.(data["px1c"])
    params[:px2c] = Float64.(data["px2c"])
    params[:px3c] = Float64.(data["px3c"])
    params[:p] = Float64(data["p"])
    
    DataParameters(;params...)
end
read_params( "tab_otjoint_00_00.json")    

@time OptimalTransportDataIntegration.unbalanced_modality(data)

data = generate_xcat_ycat(read_params( "tab_otjoint_00_00.json") )
@time OptimalTransportDataIntegration.unbalanced_modality(data)

data = CSV.read("tab_otjoint_00_00.csv", DataFrame) # generated with Python code
@time OptimalTransportDataIntegration.otjoint( data; lambda_reg = 0.0, maxrelax = 0.0, percent_closest = 0.2)

data = generate_xcat_ycat(read_params( "tab_otjoint_00_00.json") )
@time OptimalTransportDataIntegration.otjoint( data; lambda_reg = 0.0, maxrelax = 0.0, percent_closest = 0.2)

@time OptimalTransportDataIntegration.otjoint( data; lambda_reg = 0.7, maxrelax = 0.4, percent_closest = 0.2)

@time OptimalTransportDataIntegration.simple_learning( data; hidden_layer_size = 10,  learning_rate = 0.01, batchsize=64, epochs = 500)

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



# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,jl
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
# ---

using OptimalTransportDataIntegration
using DataFrames
using JSON, CSV

csv_file = joinpath(@__DIR__, "tab_otjoint_00_00.csv")
data = CSV.read(csv_file, DataFrame) # generated with Python code
@time OptimalTransportDataIntegration.unbalanced_modality(data)

@time OptimalTransportDataIntegration.unbalanced_modality(data)

json_file = joinpath(@__DIR__, "tab_otjoint_00_00.json")
read_params(json_file)

unique(data.Y), unique(data.Z)

@time OptimalTransportDataIntegration.unbalanced_modality(data)

data = CSV.read(csv_file, DataFrame) # generated with Python code
@time OptimalTransportDataIntegration.otjoint(
    data;
    lambda = 0.0,
    alpha = 0.0,
    percent_closest = 0.2,
)

data = generate_data(read_params(json_file))

@time OptimalTransportDataIntegration.otjoint(
    data;
    lambda = 0.0,
    alpha = 0.0,
    percent_closest = 0.2,
)

@time OptimalTransportDataIntegration.otjoint(
    data;
    lambda = 0.7,
    alpha = 0.4,
    percent_closest = 0.2,
)

@time OptimalTransportDataIntegration.simple_learning(
    data;
    hidden_layer_size = 10,
    learning_rate = 0.01,
    batchsize = 64,
    epochs = 500,
)

# +
using ProgressMeter

function run_simulations(simulations)

    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], p = 0.3)

    rng = PDataGenerator(params)
    prediction_quality = []

    @showprogress 1 for i = 1:simulations

        data = generate_data(rng)

        yb, za = OptimalTransportDataIntegration.unbalanced_modality(data)
        
        err1 = accuracy(data, yb, za)

        yb, za = OptimalTransportDataIntegration.otjoint(
            data;
            lambda = 0.392,
            alpha = 0.714,
            percent_closest = 0.2,
        )

        err3 = accuracy(data, yb, za)

        yb, za = OptimalTransportDataIntegration.simple_learning(
            data;
            hidden_layer_size = 10,
            learning_rate = 0.01,
            batchsize = 64,
            epochs = 500,
        )

        err3 = accuracy(data, yb, za)

        push!(prediction_quality, (err1, err2, err3))

    end

    prediction_quality

end
# -

run_simulations(10)

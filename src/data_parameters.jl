import JSON
export DataParameters
export read_params
export save_params

@with_kw struct DataParameters

    nA::Int = 1000
    nB::Int = 1000
    mA::Vector{Float64} = [0.0, 0.0, 0.0]
    mB::Vector{Float64} = [0.0, 0.0, 0.0]
    covA::Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    covB::Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    px1c::Vector{Float64} = [0.5, 0.5]
    px2c::Vector{Float64} = [0.333, 0.334, 0.333]
    px3c::Vector{Float64} = [0.25, 0.25, 0.25, 0.25]
    aA::Vector{Float64} = [1.0, 1.0, 1.5, 1, 1.5, 2]
    aB::Vector{Float64} = [1.0, 1.0, 1.5, 1, 1.5, 2]
    r2::Float64 = 0.6

end

"""
$(SIGNATURES)

Read the data generation scenario from a JSON file
"""
function read_params(jsonfile::AbstractString)

    data = JSON.parsefile(jsonfile)

    nA = Int(data["nA"])
    nB = Int(data["nB"])

    aA = Int.(data["aA"])
    aB = Int.(data["aB"])

    mA = vec(Int.(data["mA"]))
    mB = vec(Int.(data["mB"]))

    covA = stack([Float64.(x) for x in data["covA"]])
    covB = stack([Float64.(x) for x in data["covB"]])

    px1c = Float64.(data["px1c"])
    px2c = Float64.(data["px2c"])
    px3c = Float64.(data["px3c"])
    r2 = Float64(data["r2"])

    DataParameters(nA, nB, mA, mB, covA, covB, px1c, px2c, px3c, p, aA, aB, r2)

end

"""
$(SIGNATURES)

Write the data generation scenario to a JSON file
"""
function save_params(jsonfile::AbstractString, params::DataParameters)

    data = Dict(
        fieldnames(DataParameters) .=>
            getfield.(Ref(params), fieldnames(DataParameters)),
    )

    open(jsonfile, "w") do io
        JSON.print(io, data)
    end

end

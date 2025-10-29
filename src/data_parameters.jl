import JSON
export DataParameters
export read_params
export save_params

@with_kw struct DataParameters

    nA::Int = 1000
    nB::Int = 1000
    mA::Vector{Float64} = [0.0, 0.0, 0.0]
    mB::Vector{Float64} = [1.0, 1.0, 0.0]
    covA::Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    covB::Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    aA::Vector{Float64} = [1.0, 1.0, 1.5, 1, 1.5, 2]
    aB::Vector{Float64} = [1.0, 1.0, 1.5, 1, 1.5, 2]
    r2::Float64 = 0.6
    pA::Vector{Vector{Float64}} = [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]
    pB::Vector{Vector{Float64}} = [[0.8, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]

end

"""
$(SIGNATURES)

Read the data generation scenario from a JSON file
"""
function read(jsonfile::AbstractString)

    data = JSON.parsefile(jsonfile)

    nA = Int(data["nA"])
    nB = Int(data["nB"])

    aA = Int.(data["aA"])
    aB = Int.(data["aB"])

    mA = vec(Int.(data["mA"]))
    mB = vec(Int.(data["mB"]))

    covA = stack([Float64.(x) for x in data["covA"]])
    covB = stack([Float64.(x) for x in data["covB"]])

    r2 = Float64(data["r2"])
    pA = [ Float64.(v) for v in data["pA"]]
    pB = [ Float64.(v) for v in data["pB"]]

    return DataParameters(nA, nB, mA, mB, covA, covB, aA, aB, r2, pA, pB)

end

"""
$(SIGNATURES)

Write the data generation scenario to a JSON file
"""
function save(jsonfile::AbstractString, params::DataParameters)

    data = Dict(
        fieldnames(DataParameters) .=>
            getfield.(Ref(params), fieldnames(DataParameters)),
    )

    return open(jsonfile, "w") do io
        JSON.print(io, data)
    end

end

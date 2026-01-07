import JSON
export DataParameters
export read
export save

"""
    DataParameters

Scenario parameters for synthetic data generation.

Specifies all hyperparameters controlling data generation for both discrete and continuous
covariate settings. Includes sample sizes, covariate distributions, effect sizes, noise levels,
and outcome category probabilities. Can be persisted to/from JSON for reproducible experiments.

# Fields
- `nA::Int`: Sample size for base A; default: 1000
- `nB::Int`: Sample size for base B; default: 1000
- `mA::Vector{Float64}`: Mean vector for base A covariates (MVN); default: [0.0, 0.0, 0.0]
- `mB::Vector{Float64}`: Mean vector for base B covariates (MVN); default: [1.0, 1.0, 0.0]
- `covA::Matrix{Float64}`: Covariance matrix for base A covariates; default: 3×3 with diag=1, off-diag=0.2
- `covB::Matrix{Float64}`: Covariance matrix for base B covariates; default: 3×3 with diag=1, off-diag=0.2
- `aA::Vector{Float64}`: Effect sizes (regression coefficients) for base A outcomes; default: [1.0, 1.0, 1.0]
- `aB::Vector{Float64}`: Effect sizes (regression coefficients) for base B outcomes; default: [1.0, 1.0, 1.0]
- `r2::Float64`: Coefficient of determination (signal-to-noise ratio); default: 0.6 (higher = stronger signal)
- `pA::Vector{Vector{Float64}}`: Category probabilities for base A categorical covariates (one vector per dimension)
  - default: [[0.5, 0.5], [1/3, 1/3, 1/3], [0.25, 0.25, 0.25, 0.25]]
  - For discrete generator: multinomial probabilities
  - For continuous generator: unused
- `pB::Vector{Vector{Float64}}`: Category probabilities for base B categorical covariates (one vector per dimension)
  - default: [[0.8, 0.2], [1/3, 1/3, 1/3], [0.25, 0.25, 0.25, 0.25]]
  - For discrete generator: multinomial probabilities
  - For continuous generator: unused

# Usage
Create with defaults:
```julia
params = DataParameters()
```

Customize selected fields:
```julia
params = DataParameters(nA=500, nB=500, r2=0.8, mB=[2.0, 2.0, 0.0])
```

# Key Concepts
- **Covariate Shift**: Controlled by different mA/mB (continuous) or pA/pB (discrete)
- **Sample Imbalance**: Controlled by different nA/nB
- **Signal Strength**: Controlled by r2 (0 = pure noise, 1 = perfect signal)
- **Effect Sizes**: aA/aB control how strongly outcomes depend on covariates
- **Bases Design**: Base A has outcome Z, Base B has outcome Y; X covariates overlap

# See Also
- `DiscreteDataGenerator`: Generator for discrete covariate scenarios
- `ContinuousDataGenerator`: Generator for continuous covariate scenarios
- `read`: Load parameters from JSON file
- `save`: Persist parameters to JSON file

# Notes
- Dimensions: length(mA) = size(covA,1) and length(aA) = length(mA) must hold
- Probabilities in pA/pB must sum to 1.0 for each dimension
- r2 controls noise via σ²_error = (1/r² - 1) × Var(X'a)
"""
@with_kw struct DataParameters

    nA::Int = 1000
    nB::Int = 1000
    mA::Vector{Float64} = [0.0, 0.0, 0.0]
    mB::Vector{Float64} = [1.0, 1.0, 0.0]
    covA::Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    covB::Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    aA::Vector{Float64} = [1.0, 1.0, 1.0]
    aB::Vector{Float64} = [1.0, 1.0, 1.0]
    r2::Float64 = 0.6
    pA::Vector{Vector{Float64}} = [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]
    pB::Vector{Vector{Float64}} = [[0.8, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]

end

"""
    read(jsonfile::AbstractString)

Load data generation scenario parameters from a JSON file.

Deserializes a JSON file containing all `DataParameters` fields and reconstructs
a `DataParameters` instance. Useful for reproducible experiments across sessions
and for tracking scenario configurations.

# Arguments
- `jsonfile::AbstractString`: Path to JSON file containing scenario parameters

# Returns
- `DataParameters`: Reconstructed parameter object with all fields populated

# File Format
JSON structure with top-level keys matching `DataParameters` field names:
```json
{
  "nA": 1000,
  "nB": 1000,
  "mA": [0.0, 0.0, 0.0],
  "mB": [1.0, 1.0, 0.0],
  "covA": [[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]],
  "covB": [[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]],
  "aA": [1.0, 1.0, 1.0],
  "aB": [1.0, 1.0, 1.0],
  "r2": 0.6,
  "pA": [[0.5, 0.5], [0.333, 0.333, 0.333], [0.25, 0.25, 0.25, 0.25]],
  "pB": [[0.8, 0.2], [0.333, 0.333, 0.333], [0.25, 0.25, 0.25, 0.25]]
}
```

# Example
```julia
params = read("scenario_1.json")
gen = DiscreteDataGenerator(params)
data = generate(gen)
```

# See Also
- `save`: Persist parameters to JSON
- `DataParameters`: Parameter type
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
    save(jsonfile::AbstractString, params::DataParameters)

Persist data generation scenario parameters to a JSON file.

Serializes all `DataParameters` fields to a JSON file for later retrieval.
Essential for reproducible experiments: save parameters used in generation,
then load them later to recreate identical scenarios.

# Arguments
- `jsonfile::AbstractString`: Path where JSON file will be written
- `params::DataParameters`: Parameter object to serialize

# Returns
- Nothing; writes JSON file as side effect

# Example
```julia
params = DataParameters(nA=500, nB=500, r2=0.8)
save("my_scenario.json", params)

# Later...
params_loaded = read("my_scenario.json")
@assert params == params_loaded  # Exact reproduction
```

# See Also
- `read`: Load parameters from JSON
- `DataParameters`: Parameter type
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

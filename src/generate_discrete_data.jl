using Distributions
using DataFrames
import StatsBase: countmap
import OrderedCollections: OrderedDict

digitize(x, bins) = searchsortedlast.(Ref(bins), x)

to_categorical(x) = sort(unique(x)) .== permutedims(x)

to_categorical(x, levels) = levels .== permutedims(x)

export DiscreteDataGenerator

export generate

"""
    DiscreteDataGenerator

Factory for generating synthetic datasets with discrete (categorical) covariates and categorical outcomes.

Generates two data sources (base A and B) with categorical covariates (X) sampled from 
multinomial distributions and categorical outcomes (Y, Z) derived from linear combinations of 
X indicators plus noise. The generator pre-computes outcome quantiles and binning thresholds 
during construction for efficient repeated generation via `generate()`.

# Fields
- `params::DataParameters`: Scenario parameters (outcome probabilities, effect sizes, r²)
- `covA::Matrix{Float64}`: Empirical covariance matrix of covariates in base A
- `covB::Matrix{Float64}`: Empirical covariance matrix of covariates in base B
- `binsYA::Vector{Float64}`: Quantile thresholds for digitizing outcome Y in base A
- `binsZA::Vector{Float64}`: Quantile thresholds for digitizing outcome Z in base A
- `binsYB::Vector{Float64}`: Quantile thresholds for digitizing outcome Y in base B
- `binsZB::Vector{Float64}`: Quantile thresholds for digitizing outcome Z in base B

# Constructor
```julia
DiscreteDataGenerator(params; scenario=1, n=10000)
```

## Arguments
- `params::DataParameters`: Data scenario parameters including:
  - `nA`, `nB`: Sample sizes for bases A and B
  - `pA`, `pB`: Outcome category probabilities as vectors of vectors
    - `pA[1]`: Probabilities for 1st covariate categories
    - `pA[2]`: Probabilities for 2nd covariate categories, etc.
  - `aA`, `aB`: Effect size vectors (regression coefficients for covariate indicators)
  - `r2`: Coefficient of determination (controls noise level)

## Keyword Arguments
- `scenario::Int`: Determines binning strategy; default: 1
  - `scenario=1`: Same binning thresholds for A and B (no covariate shift)
  - `scenario≠1`: Separate binning thresholds per base (covariate shift)
- `n::Int`: Unused parameter (kept for API compatibility); default: 10000

# Generation Process (Constructor)
1. Sample categorical covariates: XA[i,j] ~ Categorical(pA[j]) for each dimension
2. Compute empirical covariances of sampled covariates
3. Compute error variance from r² and effect sizes: σ²_error = (1/r² - 1) × ∑(aᵢ aⱼ Cov[i,j])
4. Generate latent outcomes: Y = X'a + ε with ε ~ N(0, σ²_error) (when σ² > 0)
5. Compute quantiles at [0.25, 0.5, 0.75] for Y and [1/3, 2/3] for Z
6. Create bins: [-∞, q₁, q₂, q₃, ∞] (4 categories) and [-∞, q₁, q₂, ∞] (3 categories)

# See Also
- `ContinuousDataGenerator`: For continuous (multivariate normal) covariates
- `generate`: Call this method on generator to produce DataFrame

# Notes
- Covariates are categorical (sampled from multinomial distributions)
- Outcomes are always categorical (3-4 classes based on quantile binning)
- R² parameter controls signal-to-noise ratio in linear model
- Scenario parameter allows testing covariate shift and distribution mismatch assumptions
- Covariance matrices computed from sampled data may vary between generator instances
"""
struct DiscreteDataGenerator

    params::DataParameters
    covA::Matrix{Float64}
    covB::Matrix{Float64}
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}

    function DiscreteDataGenerator(params; scenario = 1, n = 10000)

        q = length(params.mA)

        XA = stack([rand(Categorical(params.pA[i]), params.nA) for i in 1:q], dims = 1)
        XB = stack([rand(Categorical(params.pB[i]), params.nB) for i in 1:q], dims = 1)

        X1 = XA
        X2 = XB

        if q === 1
            covA = fill(cov(vec(XA)), (1, 1))
            covB = fill(cov(vec(XB)), (1, 1))
        else
            covA = cov(XA, dims = 1)
            covB = cov(XB, dims = 1)
        end

        aA = params.aA
        aB = params.aB

        cr2 = 1 / params.r2 - 1

        σA = cr2 * sum([params.aA[i] * aA[j] * cov(XA[i, :], XA[j, :]) for i in 1:q, j in 1:q])
        σB = cr2 * sum([params.aB[i] * aB[j] * cov(XB[i, :], XB[j, :]) for i in 1:q, j in 1:q])

        if σA == 0
            Base1 = X1' * params.aA[1:q]  # pas de bruit
        else
            Base1 = X1' * params.aA[1:q] .+ rand(Normal(0.0, sqrt(σA)), params.nA)
        end

        if σB == 0
            Base2 = X2' * params.aB[1:q]  # pas de bruit
        else
            Base2 = X2' * params.aB[1:q] .+ rand(Normal(0.0, sqrt(σB)), params.nB)
        end

        bYA = quantile(Base1, [0.25, 0.5, 0.75])
        bYB = quantile(Base2, [0.25, 0.5, 0.75])
        bZA = quantile(Base1, [1 / 3, 2 / 3])
        bZB = quantile(Base2, [1 / 3, 2 / 3])

        if scenario == 1
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = copy(binsYA)
            binsZB = copy(binsZA)
        else
            binsYA = vcat(-Inf, bYA, Inf)
            binsZA = vcat(-Inf, bZA, Inf)
            binsYB = vcat(-Inf, bYB, Inf)
            binsZB = vcat(-Inf, bZB, Inf)
        end

        return new(
            params,
            covA,
            covB,
            binsYA,
            binsZA,
            binsYB,
            binsZB,
        )

    end

end


"""
    generate(generator::DiscreteDataGenerator; eps=0.0)

Generate a synthetic dataset with discrete (categorical) covariates and categorical outcomes.

Samples new categorical covariates from multinomial distributions specified in generator,
computes latent outcomes via linear model on covariate indicators, then discretizes into 
categorical bins pre-computed during generator construction.

# Arguments
- `generator::DiscreteDataGenerator`: Pre-configured generator with binning thresholds and covariance estimates

# Keyword Arguments
- `eps::Float64`: Offset added to outcome Z binning thresholds (for covariate shift sensitivity); default: 0.0

# Returns
- `DataFrame`: Dataset with columns:
  - `X1, X2, ...`: Categorical covariates (dimension matches `length(params.pA)`)
  - `Y`: Categorical outcome (1:4) indicating base B's observed outcome
  - `Z`: Categorical outcome (1:3) indicating base A's observed outcome
  - `database`: Integer (1 for base A, 2 for base B) indicating data source
  - Total rows: `params.nA + params.nB`

# Algorithm
1. Sample categorical covariates: XA[i,j] ~ Categorical(pA[j]) for each base independently
2. Compute latent outcomes: Y1 = XA'·aA + ε_A and Y2 = XB'·aB + ε_B
3. Digitize into categories using pre-computed bins:
   - YA = digitize(Y1, binsYA), ZA = digitize(Y1, binsZA)
   - YB = digitize(Y2, binsYB + eps), ZB = digitize(Y2, binsZB + eps)
4. Assemble DataFrame with database indicator
5. Log category distributions (info level)

# Details
- **Data split**: Base A (database=1) has outcomes Z, Base B (database=2) has outcomes Y
- **Missing outcomes**: Implicit in database indicator (no NAs in columns)
- **Outcome relationship**: Z and Y in same base are derived from same latent variable
- **eps parameter**: Small perturbation useful for testing robustness to binning threshold changes
- **Covariate sampling**: Independent samples each call, so distributions may vary slightly from params

# Example
```julia
params = DataParameters(nA=500, nB=500, r2=0.6)
gen = DiscreteDataGenerator(params, scenario=1)
data = generate(gen)  # 1000 rows × 5 columns (3 covariates + Y + Z + database)
```

# See Also
- `DiscreteDataGenerator`: Constructor with binning logic
- `ContinuousDataGenerator` / `generate`: Continuous covariates alternative
"""
function generate(generator::DiscreteDataGenerator; eps = 0.0)

    params = generator.params

    q = length(params.mA)

    XA = stack([rand(Categorical(params.pA[i]), params.nA) for i in 1:q], dims = 1)
    XB = stack([rand(Categorical(params.pB[i]), params.nB) for i in 1:q], dims = 1)

    X1 = XA
    X2 = XB

    cr2 = 1.0 / params.r2 - 1

    aA = params.aA[1:q]
    aB = params.aB[1:q]

    covA = generator.covA
    covB = generator.covB

    cr2 = 1 / params.r2 - 1
    σA = cr2 * sum([aA[i] * aA[j] * cov(XA[i, :], XA[j, :]) for i in 1:q, j in 1:q])
    σB = cr2 * sum([aB[i] * aB[j] * cov(XB[i, :], XB[j, :]) for i in 1:q, j in 1:q])

    if σA == 0
        Y1 = X1' * aA
    else
        Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(σA)), params.nA)
    end

    if σB == 0
        Y2 = X2' * aB
    else
        Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(σB)), params.nB)
    end

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)

    for j in 1:q
        @info "Categories in XA$j $(sort!(OrderedDict(countmap(XA[j, :]))))"
        @info "Categories in XB$j $(sort!(OrderedDict(countmap(XB[j, :]))))"
    end

    colnames = Symbol.("X" .* string.(1:q))

    df = DataFrame(hcat(X1, X2)', colnames)

    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))" #
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

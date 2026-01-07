export ContinuousDataGenerator

"""
    ContinuousDataGenerator

Factory for generating synthetic datasets with continuous covariates and categorical outcomes.

Generates two data sources (base A and B) with multivariate normal covariates (X) and 
categorical outcomes (Y, Z) derived from linear combinations of X plus noise. The generator 
pre-computes outcome quantiles and binning thresholds during construction for efficient 
repeated generation via `generate()`.

# Fields
- `params::DataParameters`: Scenario parameters (means, covariances, effect sizes, outcome probabilities)
- `binsYA::Vector{Float64}`: Quantile thresholds for digitizing outcome Y in base A
- `binsZA::Vector{Float64}`: Quantile thresholds for digitizing outcome Z in base A
- `binsYB::Vector{Float64}`: Quantile thresholds for digitizing outcome Y in base B
- `binsZB::Vector{Float64}`: Quantile thresholds for digitizing outcome Z in base B

# Constructor
```julia
ContinuousDataGenerator(params; scenario=1, n=10000)
```

## Arguments
- `params::DataParameters`: Data scenario parameters including:
  - `nA`, `nB`: Sample sizes for bases A and B
  - `mA`, `mB`: Mean vectors for multivariate normal covariates
  - `covA`, `covB`: Covariance matrices for covariates
  - `aA`, `aB`: Effect size vectors (regression coefficients on X)
  - `r2`: Coefficient of determination (controls noise level)
  - `pA`, `pB`: Outcome category probabilities (unused for continuous generation)

## Keyword Arguments
- `scenario::Int`: Determines binning strategy; default: 1
  - `scenario=1`: Same binning thresholds for A and B (no covariate shift)
  - `scenario≠1`: Separate binning thresholds per base (covariate shift)
- `n::Int`: Unused parameter (kept for API compatibility); default: 10000

# Generation Process (Constructor)
1. Sample covariates: XA ~ N(mA, covA) and XB ~ N(mB, covB)
2. Compute error variance from r² and effect sizes: σ²_error = (1/r² - 1) × ∑(aᵢ aⱼ Cov[i,j])
3. Generate latent outcomes: Y = X'a + ε with ε ~ N(0, σ²_error)
4. Compute quantiles at [0.25, 0.5, 0.75] for Y and [1/3, 2/3] for Z
5. Create bins: [-∞, q₁, q₂, q₃, ∞] (4 categories) and [-∞, q₁, q₂, ∞] (3 categories)

# See Also
- `DiscreteDataGenerator`: For categorical covariates
- `generate`: Call this method on generator to produce DataFrame

# Notes
- Covariates are continuous (multivariate normal)
- Outcomes are always categorical (3-4 classes based on quantile binning)
- R² parameter controls signal-to-noise ratio in linear model
- Scenario parameter allows testing covariate shift and distribution mismatch assumptions
"""
struct ContinuousDataGenerator

    params::DataParameters
    binsYA::Vector{Float64}
    binsZA::Vector{Float64}
    binsYB::Vector{Float64}
    binsZB::Vector{Float64}

    function ContinuousDataGenerator(params; scenario = 1, n = 10000)

        XA = rand(MvNormal(params.mA, params.covA), params.nA)
        XB = rand(MvNormal(params.mB, params.covB), params.nB)

        X1 = XA
        X2 = XB

        qA = size(XA, 1)
        qB = size(XB, 1)

        aA = params.aA[1:qA]
        aB = params.aB[1:qB]

        cr2 = 1 / params.r2 - 1

        covA = params.covA
        covB = params.covB

        varerrorA =
            cr2 *
            sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
        varerrorB =
            cr2 *
            sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])

        if varerrorA == 0
            Base1 = X1' * aA
        else
            Base1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
        end

        if varerrorB == 0
            Base2 = X2' * aB
        else
            Base2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
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
            binsYA,
            binsZA,
            binsYB,
            binsZB,
        )


    end

end


"""
    generate(generator::ContinuousDataGenerator; eps=0.0)

Generate a synthetic dataset with continuous covariates and categorical outcomes.

Samples new covariates from the multivariate normal distributions specified in generator,
computes latent outcomes via linear model, then discretizes into categorical bins pre-computed
during generator construction.

# Arguments
- `generator::ContinuousDataGenerator`: Pre-configured generator with binning thresholds

# Keyword Arguments
- `eps::Float64`: Offset added to outcome Z binning thresholds (for covariate shift sensitivity); default: 0.0

# Returns
- `DataFrame`: Dataset with columns:
  - `X1, X2, ...`: Continuous covariates (dimension matches `params.mA` length)
  - `Y`: Categorical outcome (1:4) indicating base B's observed outcome
  - `Z`: Categorical outcome (1:3) indicating base A's observed outcome
  - `database`: Integer (1 for base A, 2 for base B) indicating data source
  - Total rows: `params.nA + params.nB`

# Algorithm
1. Sample covariates: XA ~ N(mA, covA) and XB ~ N(mB, covB)
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

# Example
```julia
params = DataParameters(nA=500, nB=500, r2=0.6)
gen = ContinuousDataGenerator(params)
data = generate(gen)  # 1000 rows × 5 columns
```

# See Also
- `ContinuousDataGenerator`: Constructor with binning logic
- `DiscreteDataGenerator` / `generate`: Discrete covariates alternative
"""
function generate(generator::ContinuousDataGenerator; eps = 0.0)

    params = generator.params

    XA = rand(MvNormal(params.mA, params.covA), params.nA)
    XB = rand(MvNormal(params.mB, params.covB), params.nB)
    X1 = XA
    X2 = XB

    cr2 = 1.0 / params.r2 - 1

    qA = size(XA, 1)
    qB = size(XB, 1)

    aA = params.aA[1:qA]
    aB = params.aB[1:qB]

    covA = params.covA
    covB = params.covB

    varerrorA =
        cr2 *
        sum([aA[i] * aA[j] * covA[i, j] for i in axes(covA, 1), j in axes(covA, 2)])
    varerrorB =
        cr2 *
        sum([aB[i] * aB[j] * covB[i, j] for i in axes(covB, 1), j in axes(covB, 2)])


    if varerrorA == 0
        Y1 = X1' * aA
    else
        Y1 = X1' * aA .+ rand(Normal(0.0, sqrt(varerrorA)), params.nA)
    end

    if varerrorB == 0
        Y2 = X2' * aB
    else
        Y2 = X2' * aB .+ rand(Normal(0.0, sqrt(varerrorB)), params.nB)
    end

    YA = digitize(Y1, generator.binsYA)
    ZA = digitize(Y1, generator.binsZA)

    YB = digitize(Y2, generator.binsYB .+ eps)
    ZB = digitize(Y2, generator.binsZB .+ eps)

    p = length(aA)
    colnames = Symbol.("X" .* string.(1:p))
    df = DataFrame(vcat(X1', X2'), colnames)
    df.Y = vcat(YA, YB)
    df.Z = vcat(ZA, ZB)
    df.database = vcat(fill(1, params.nA), fill(2, params.nB))

    @info "Categories in YA $(sort!(OrderedDict(countmap(YA))))"
    @info "Categories in ZA $(sort!(OrderedDict(countmap(ZA))))"
    @info "Categories in YB $(sort!(OrderedDict(countmap(YB))))"
    @info "Categories in ZB $(sort!(OrderedDict(countmap(ZB))))"

    return df

end

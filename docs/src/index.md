# Optimal Transport for Statistical Matching

```@meta
CurrentModule = OptimalTransportDataIntegration
```

**Authors**: Valérie Garès, Jeremy Omer and Pierre Navaro.

**Affiliations**: IRMAR (Université de Rennes, INRIA, CNRS, INSA Rennes)

**Objective**: Integrate distinct datasets on the same population, sharing some variables but containing unique ones, to create a unified synthetic dataset.

**Proposed Methods**:
- Two approaches based on **Optimal Transport (OT)** theory:
  1. [`JointOTWithinBase`](@ref): Optimal transport of joint distributions of explanatory variables and outcomes **within** each data source.
  2. [`JointOTBetweenBases`](@ref): Optimal transport of joint distributions **between** data sources, with simultaneous estimation of predictive functions.

## 🔍 Introduction

### Context

- **Data Integration**: Combine heterogeneous data to improve the reliability of statistical analyses.
- **Statistical Matching (SM)**: Merge datasets with **distinct units** but shared/common variables.
- **Challenge**: Estimate relationships between variables not jointly observed (e.g., `Y` and `Z` conditional on `X`).

### Formalization

| Dataset       | covariables        | outcomes             |                      |
|---------------|--------------------|----------------------|----------------------|
| Dataset ``A`` | ``X^A`` (observed) | ``Y^A`` (observed)   | ``Z^A`` (unobserved) |
| Dataset ``B`` | ``X^B`` (observed) | ``Y^B`` (unobserved) | ``Z^B`` (observed)   |

**Goal**: Build a synthetic dataset where ``Y`` and ``Z`` are predicted for all units.

---

## 🛠️ Methods

### 1. JointOTWithinBase (JDOT-wi): Optimal Transport **Within** Datasets

- **Assumption**: Conditional distributions ``Y|X`` and ``Z|X`` are identical between ``A`` and ``B``.

- **Algorithm**:
   1. Estimate joint distributions ``\mu^{(X^A,Y^A)}`` and ``\mu^{(X^A,Z^A)}`` using empirical estimators.
   2. Solve an optimal transport problem to approximate ``\mu^{(X^A,Y^A,Z^A)}``.
   3. Predict ``Z^A`` and ``Y^B`` using optimized functions ``f`` and ``g``.

- **Advantages**:

  - Suitable for categorical variables.
  - Robust if the conditional distribution assumption holds.

- **Limitations**:

  - Requires discretization for continuous variables.
  - Sensitive to violations of the conditional distribution assumption.

---

### 2. JointOTBetweenBases (JDOT-be): Optimal Transport **Between** Datasets

- **Approach**: Transport the joint distribution ``(X, Y, Z)`` between ``A`` and ``B`` by simultaneously optimizing:
  - A transport plan ``\gamma``.
  - Predictive functions ``f`` (for ``Z^A``) and ``g`` (for ``Y^B``).

- **Cost Function**:

```math

c\big((x,y,z),(x',y',z')\big) = d(x,x') + \alpha_1 \mathcal{L}_Y(y,y') + \alpha_2 \mathcal{L}_Z(z,z')

```

**Optimization**: Alternate between:
1. Updating the transport plan ``\gamma``.
2. Optimizing functions ``f`` and ``g`` (e.g., neural networks).

**Advantages**:
- Handles imbalanced data via unbalanced optimal transport.
- Suitable for mixed data types (continuous/categorical).
- No strong CIA (Conditional Independence Assumption) required.

**Limitations**:
- High computational cost.
- Sensitive to hyperparameters (``\alpha_1, \alpha_2, \rho``).

---

## First example

### Installation

Launch the Julia REPL and enter the Pkg REPL by pressing `]`. To get back to the Julia REPL, press the backspace.

Upon entering the Pkg REPL, you should see the following prompt:

```julia-repl
(@v1.12) pkg>
```

To install the package from GitHub, use `add`:

```julia-repl
(@v1.12) pkg> add https://github.com/otrecoding/OptimalTransportDataIntegration.jl.git
     Cloning git-repo `https://github.com/otrecoding/OptimalTransportDataIntegration.jl.git`
    Updating git-repo `https://github.com/otrecoding/OptimalTransportDataIntegration.jl.git`
   Resolving package versions...
   Installed PythonOT ─ v0.1.6
    Updating `~/tmp/Project.toml`
```

After the package is installed, you need install POT the Python Optimal Transport library

```julia-repl
julia> using Conda
 │ Package Conda not found, but a package named Conda is available from a registry.
 │ Install package?
 │   (tmp) pkg> add Conda
 └ (y/n/o) [y]:
   Resolving package versions...
    Updating `~/tmp/Project.toml`
  [8f4d0f93] + Conda v1.10.3
    Manifest No packages added to or removed from `~/tmp/Manifest.toml`

julia> Conda.add("pot")
[ Info: Running `conda install -y pot` in root environment
Retrieving notices: done
Channels:
 - conda-forge
Platform: osx-64
Collecting package metadata (repodata.json): \
```

Now the package can be loaded into the Julia session:

```julia-repl
julia> using OptimalTransportDataIntegration

julia>
```

### Parameters

Create some defaults paramaters with [`DataParameters`](@ref).


```julia-repl
julia> params = DataParameters()
```

### Generators

You can generate continuous covariables ``X`` with [`ContinuousDataGenerator`](@ref) or
discrete covariables ``X`` with [`DiscreteDataGenerator`](@ref).

- Two scenarios:
  - **`scenario = 1`**: ``\mathbb{P}_{Z^A|X^A=x} = \mathbb{P}_{Z^B|X^B=x}`` (identical conditional distributions).
  - **`scenario = 2`**: ``\mathbb{P}_{Z^A|X^A=x} = \mathbb{P}_{Z^B|X^B=T(x)}`` (transformation ``T`` between domains).

```julia

rng = ContinuousDataGenerator(params)
data = generate(rng)

result = otrecod(data, JointOTWithinBase())
println(" within-r : $(accuracy(result)) ")

result = otrecod(data, JointOTBetweenBases())
println(" between-r : $(accuracy(result))")

result = otrecod(data, SimpleLearning())
println(" sl : $(accuracy(result))")
```

The simulator provided by the package generates a dataframe containing
the covariates ``X`` from databases ``A`` and ``B`` and the outcomes ``Y`` from
database ``A`` and ``Z`` from database ``B``. The [`otrecod`](@ref) function can be used
to obtain a recoded version of variable ``Y`` for database ``B`` and ``Z`` for
database ``A``. The [`accuracy`](@ref) function can be used to verify the quality
of the method. It gives you the error on ``Y``, ``Z`` and both outcomes.

Methods available as `otrecod` second argument are listed below with their paper names from
[*Optimal transport for statistical matching*](https://hal.science/hal-05517603):

- [`JointOTWithinBase`](@ref): Regularized OT matching of covariate-outcome distributions within each dataset. **Paper:** JDOT-wi-r.
- [`SimpleLearning`](@ref): Two supervised neural networks predict each missing outcome from the observed covariates. **Paper:** Simple learning.
- [`JointOTBetweenBases`](@ref): Regularized unbalanced OT matching of covariate-outcome distributions between datasets. **Paper:** JDOT-be-r-un.
- [`JointOTBetweenBasesDiscreteOrdered`](@ref): Categorical-outcome variant of between-bases OT, using one-hot outcomes and prediction-loss refinement. **Paper:** no separate name; an implementation variant of JDOT-be-r-un.
- [`JointOTBetweenBasesWithPredictors`](@ref): Alternates between transport and neural-network prediction using covariates and observed outcomes. **Paper:** JDOT-be-xyz.
- [`JointOTBetweenBasesJDOT`](@ref): Jointly learns outcome predictors and two outcome-specific transport plans between datasets. **Paper:** JDOT-base.
- [`JointOTDABetweenBasesCovariables`](@ref): Matches covariate distributions with OT, then trains predictors on transported covariates. **Paper:** OTDA baryFit.
- [`JointOTDABetweenBasesOutcomes`](@ref): Transports outcome class proportions with a covariate-based OT plan. **Paper:** OTDA labelProp.
- [`JointOTDABetweenBasesOutcomesWithPredictors`](@ref): Transports outcome labels with OT, then fits neural-network predictors. **Paper:** OTDA labelFit.
- [`JointOTBetweenBasesWithoutOutcomes`](@ref): Matches covariates and learns outcome predictors from covariates alone. **Paper:** JDOT-be with ``Y=f(X)`` and ``Z=g(X)``.

## Simulation data generation

```@autodocs
Modules = [OptimalTransportDataIntegration]
Pages=["data_parameters.jl", "generate_continuous_data.jl", "generate_discrete_data.jl"]
```

## Recoding methods

```@autodocs
Modules = [OptimalTransportDataIntegration]
Pages=["otrecod.jl"]
```

## Helper functions

```@docs
accuracy
```

```@docs
confusion_matrix
```

```@docs
compute_pred_error!
```

```@index
```

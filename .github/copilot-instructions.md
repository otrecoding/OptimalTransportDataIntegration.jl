# Copilot Instructions for OptimalTransportDataIntegration.jl

## Project Overview

OptimalTransportDataIntegration.jl is a Julia package implementing
statistical data matching based on optimal transport theory. It
integrates multiple data sources (bases A and B) sharing common
covariates but with distinct outcome variables, producing a unified
dataset with all variables jointly available.

**Key domain concepts:**
- **Base A & Base B**: Two data sources with overlapping covariates (X) but different outcomes (Y for base B, Z for base A)
- **Statistical Matching**: Predicting missing outcomes using optimal transport-based methods
- **Accuracy**: Primary metric measuring prediction correctness across all outcomes

## Architecture Overview

### Core Pipeline: Data → Method → Results

1. **Data Generation** (`src/generate_*.jl`): Create synthetic datasets with controlled covariate distributions
   - `DiscreteDataGenerator` / `ContinuousDataGenerator`: Parameters-based factory using `DataParameters`
   - **Pattern**: Generator holds parameters; `generate()` method produces DataFrames with columns: `database` (1 or 2), `X1`, `X2`, ..., `Y`, `Z`

2. **Instance Creation** (`src/instance.jl`): Pre-compute distance matrices and index structures for optimization
   - Used by OT-between methods to cache expensive distance calculations between base pairs
   - Pre-filters subjects by covariate values to avoid redundant computations

3. **OT Methods** (`src/joint_ot_*.jl`): Implement statistical matching algorithms
   - **Within-base methods**: Balance distributions within single data source
   - **Between-bases methods**: Transport distributions across bases + predict missing variables
   - **discrete vs. continuous**: Auto-selected by checking if X values are integers

4. **Accuracy Computation** (`src/OptimalTransportDataIntegration.jl`): Compare predictions to ground truth
   - Returns triple: `(accuracy_yb, accuracy_za, overall_accuracy)`
   - Filters truth values by database indicator before comparison

### Method Hierarchy (Abstract Type Pattern)

All methods inherit from `AbstractMethod` and dispatch through `otrecod(data::DataFrame, method::AbstractMethod)`:

- `JointOTWithinBase`: Parameters `lambda`, `alpha`, `percent_closest`, `distance`
- `SimpleLearning`: Neural network approach with Flux; parameters `hidden_layer_size`, `learning_rate`, `batchsize`, `epochs`
- `JointOTBetweenBases`: Main OT-based method; parameters `reg`, `reg_m1`, `reg_m2`, `Ylevels`, `Zlevels`
- `JointOTBetweenBasesCategory`: Categorical outcome variant

The dispatcher checks for data type (discrete/continuous) and calls appropriate implementation (`_discrete` / `_continuous` suffixed functions).

## Key Patterns & Conventions

### Data Structure Convention

All methods receive `DataFrames` with:
- **database**: Integer (1 for A, 2 for B) indicating source
- **X-columns**: Covariates (named `X1`, `X2`, etc.)
- **Y, Z**: Outcomes (always present; missing data implicit via database indicator)
- Example: See `test/data_good.csv` and `test/data_bad.csv`

### One-Hot Encoding for Categorical Data

Use `one_hot_encoder(data::Vector, levels::Vector)` to convert categorical outcomes to matrix form.
- Required for methods comparing distributions via optimal transport (cross-entropy loss)
- See `src/joint_ot_between_bases_category.jl` for usage pattern

### Optimization Solver Integration

- **JuMP + Clp**: Linear programming for transport plans; see `joint_ot_between_bases_discrete.jl`
- **Python OT library**: Wrapped via `PythonOT.jl`; used for entropic-regularized Wasserstein
- Non-obvious: `PythonOT` is a custom package, built from https://github.com/pnavaro/PythonOT.jl

### Regularization Parameters

The `reg`, `reg_m1`, `reg_m2` parameters in `JointOTBetweenBases` control solution complexity:
- `reg`: Main entropy regularization (0 = exact OT, larger = more relaxed)
- `reg_m1`, `reg_m2`: Marginal constraints relaxation; set to 0 for balanced (harder) problems

## Developer Workflow

### Running Tests

```bash
julia --project test/runtests.jl
```
Tests use synthetic data (discrete) and CSV files. Slow tests are marked with `@time` for profiling.

### Running Examples

```bash
julia --project examples/quickstart.jl
```
Examples iterate over method variants to compare accuracy. Data parameters (means, covariances, outcome probabilities) are tuned in each example.

### Adding a New OT Method

1. Define new `struct YourMethod <: AbstractMethod` with hyperparameters in `src/otrecod.jl`
2. Create `function otrecod(data::DataFrame, method::YourMethod)` dispatcher
3. Implement separate `_your_method_discrete()` and `_your_method_continuous()` functions in new file
4. **Auto-dispatch by data type**: Dispatcher checks `all(isinteger.(X))` and routes accordingly
5. Return `JointOTResult(yb_true, za_true, yb_pred, za_pred)` tuple
6. Add test case in `test/runtests.jl` with `@testset`

### Working with Large Datasets

- Experiments use synthetic data scaled to n=10,000+
- Distance matrix computation is the bottleneck; `Instance` pre-caches for between-bases methods
- Example: `experiments/sample_size_effect.jl` compares scalability across different n values

## Testing & Quality

- Uses `Aqua.jl` for code quality checks (no stale type parameters, ambiguous methods)
- Uses `Documenter.jl` for API documentation; see `docs/src/` for markdown content
- Code coverage tracked via CodeCov CI
- All PRs require passing tests and Aqua QA before merge

## Key Files Reference

- **Entry point**: `src/OptimalTransportDataIntegration.jl` (module definition, exports)
- **Method implementations**: `src/joint_ot_between_bases_*.jl` (multiple variants for different data types/constraints)
- **Simple learning**: `src/simple_learning.jl` (Flux neural network baseline)
- **Data generation**: `src/generate_discrete_data.jl`, `src/generate_continuous_data.jl`
- **Test data**: `test/data_good.csv` (high signal), `test/data_bad.csv` (low signal)

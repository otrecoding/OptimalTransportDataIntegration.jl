# Continuous data

## Scenarios 1 and 2

Scenario 1 uses the same outcome-category bin thresholds for both databases, while scenario 2 computes thresholds separately from each database's outcomes.

- **Sample ratio** ([`sample_ratio_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/sample_ratio_effect.jl)): `nA=1000`, with `nB=nA/r` and `r=1,2,5,10`.
- **Sample size** ([`sample_size_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/sample_size_effect.jl)): `nA=nB=100,1000,5000`.
- **Covariate link** ([`covariates_link_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/covariates_link_effect.jl)): `R2=0.2,0.4,0.6,0.8,1.0`.
- **Covariate shift** ([`covariates_shift_assumption.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/covariates_shift_assumption.jl)): `mB=(0,0,0),(1,0,0),(1,1,0),(1,1,1)`.

### Methods

- Simple learning (`SimpleLearning`)
- OT within (`JointOTWithinBase`, `alpha=lambda=0`)
- Regularized OT within (`JointOTWithinBase` with parameters from `best_parameters`)
- Balanced OT between bases with predictors (`JointOTBetweenBasesWithPredictors`, `reg=0`)
- Regularized OT between bases with predictors (`JointOTBetweenBasesWithPredictors` with parameters from `best_parameters`)

### Additional experiments

- [`best_parameters.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/best_parameters.jl) searches `alpha=0:0.1:2`, `lambda=0:0.1:1`, `reg=0.001,0.01,0.1,1.0`, and `reg_m=0.01,0.05,0.1,0.25,0.5,0.75` for both data types and scenarios.
- [`best_reg_m.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/best_reg_m.jl) tests `reg_m=0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05` with `reg=0.001`, `nA=nB=500`, and the four `mB` shifts above in continuous scenario 1.
- [`reference_methods_continuous.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/reference_methods_continuous.jl) compares `SimpleLearning`, `JointOTWithinBase`, `JointOTBetweenBasesWithPredictors`, `JointOTBetweenBasesWithoutOutcomes`, `JointOTBetweenBasesJDOT`, `JointOTDABetweenBasesCovariables`, `JointOTDABetweenBasesOutcomes`, and `JointOTDABetweenBasesOutcomesWithPredictors`.
- [`computing_time.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/computing_time.jl) times five within-base, between-bases, and simple-learning variants for both data types and scenarios; it runs one simulation.

## Discrete data

### Scenarios 1 and 2

Scenario 1 uses the same outcome-category bin thresholds for both databases, while scenario 2 computes thresholds separately from each database's outcomes.

- **Sample ratio** ([`sample_ratio_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/sample_ratio_effect.jl)): `nA=1000`, with `nB=nA/r` and `r=1,2,5,10`.
- **Sample size** ([`sample_size_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/sample_size_effect.jl)): `nA=nB=100,1000,5000`.
- **Covariate link** ([`covariates_link_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/covariates_link_effect.jl)): `R2=0.2,0.4,0.6,0.8,1.0`.
- **Covariate shift** ([`covariates_shift_assumption.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/covariates_shift_assumption.jl)):
  1. `pB=pA`
  2. `pB_1=(0.8,0.2), pB_2=pA_2, pB_3=pA_3`
  3. `pB_1=(0.8,0.2), pB_2=(0.6,0.2,0.2), pB_3=pA_3`
  4. `pB_1=(0.8,0.2), pB_2=(0.6,0.2,0.2), pB_3=(0.6,0.2,0.1,0.1)`

### Methods

- Simple learning (`SimpleLearning`)
- OT within (`JointOTWithinBase`, `alpha=lambda=0`)
- Regularized OT within (`JointOTWithinBase` with parameters from `best_parameters`)
- Balanced OT between bases (`JointOTBetweenBases` or `JointOTBetweenBasesDiscreteOrdered`, `reg=0`)
- Regularized OT between bases (`JointOTBetweenBases` or `JointOTBetweenBasesDiscreteOrdered` with parameters from `best_parameters`)

### Additional experiments

- [`conditional_distribution.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/conditional_distribution.jl) varies `epsilon=0,1,2,3,5` in scenario 1. The discrete experiment is run by default; the continuous invocation is commented out.
- `tune_alpha_weights.jl` selects `alpha1` and `alpha2` separately for scenarios 1 and 2, using mean overall prediction accuracy across simulated **discrete** validation datasets. It evaluates all pairs in `[0.001, 0.01, 0.1, 1, 10]`, using the same datasets and predictor-training random seeds for each pair. The method is `JointOTBetweenBasesWithPredictors(reg = 0.0)`, with 10 iterations and its default network parameters, including 1000 training epochs. Ties select the first pair in grid order. The full search is computationally expensive.

  Run `julia --project experiments/tune_alpha_weights.jl [nsimulations] [validation_seed] [sample_size]`; defaults are 100, 20261123, and 1000 per base. All candidate scores and the chosen pair are printed for each scenario. The validation seeds are `validation_seed` and `validation_seed + 1`; generator construction and dataset generation both use those seeds.

  These scores are **not final test results**. For final evaluation, construct new generators and generate new datasets with separate seeds that differ from both validation seeds; never reuse validation datasets. Retain the selected pair for its scenario and the same method configuration. The script does not perform final evaluation.

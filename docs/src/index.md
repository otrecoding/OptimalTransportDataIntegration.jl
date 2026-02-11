# Optimal Transport for Statistical Matching

```@meta
CurrentModule = OptimalTransportDataIntegration
```

**Authors**: Chloé Friguet, Ioana Gavra, Nathalie Costet, Nicolas Courty, Pierre Navaro, Valérie Garès
**Affiliations**: IRMAR, IRISA (Université de Rennes, INRIA, CNRS, Inserm)
**Contact**: [valerie.gares@inria.fr](mailto:valerie.gares@inria.fr)

**Objective**: Integrate distinct datasets on the same population, sharing some variables but containing unique ones, to create a unified synthetic dataset.

**Proposed Methods**:
- Two approaches based on **Optimal Transport (OT)** theory:
  1. **JDOT-wi**: Optimal transport of joint distributions of explanatory variables and outcomes **within** each data source.
  2. **JDOT-be**: Optimal transport of joint distributions **between** data sources, with simultaneous estimation of predictive functions.

## 🔍 Introduction
### Context
- **Data Integration**: Combine heterogeneous data to improve the reliability of statistical analyses.
- **Statistical Matching (SM)**: Merge datasets with **distinct units** but shared/common variables.
- **Challenge**: Estimate relationships between variables not jointly observed (e.g., \(Y\) and \(Z\) conditional on \(X\)).

### Formalization
| Dataset \(A\) | \(X^A\) (observed) | \(Y^A\) (observed) | \(Z^A\) (unobserved) |
|---------------|---------------------|---------------------|-----------------------|
| Dataset \(B\) | \(X^B\) (observed) | \(Y^B\) (unobserved) | \(Z^B\) (observed) |

**Goal**: Build a synthetic dataset where \(Y\) and \(Z\) are predicted for all units.

---

## 🛠️ Methods

### 1. JDOT-wi: Optimal Transport **Within** Datasets
**Assumption**: Conditional distributions \(Y|X\) and \(Z|X\) are identical between \(A\) and \(B\).
**Algorithm**:
1. Estimate joint distributions \(\mu^{(X^A,Y^A)}\) and \(\mu^{(X^A,Z^A)}\) using empirical estimators.
2. Solve an optimal transport problem to approximate \(\mu^{(X^A,Y^A,Z^A)}\).
3. Predict \(Z^A\) and \(Y^B\) using optimized functions \(f\) and \(g\).

**Advantages**:
- Suitable for categorical variables.
- Robust if the conditional distribution assumption holds.

**Limitations**:
- Requires discretization for continuous variables.
- Sensitive to violations of the conditional distribution assumption.

---

### 2. JDOT-be: Optimal Transport **Between** Datasets
**Approach**: Transport the joint distribution \((X, Y, Z)\) between \(A\) and \(B\) by simultaneously optimizing:
- A transport plan \(\gamma\).
- Predictive functions \(f\) (for \(Z^A\)) and \(g\) (for \(Y^B\)).

**Cost Function**:
\[
c\big((x,y,z),(x',y',z')\big) = d(x,x') + \alpha_1 \mathcal{L}_Y(y,y') + \alpha_2 \mathcal{L}_Z(z,z')
\]

**Optimization**: Alternate between:
1. Updating the transport plan \(\gamma\).
2. Optimizing functions \(f\) and \(g\) (e.g., neural networks).

**Advantages**:
- Handles imbalanced data via unbalanced optimal transport.
- Suitable for mixed data types (continuous/categorical).
- No strong CIA (Conditional Independence Assumption) required.

**Limitations**:
- High computational cost.
- Sensitive to hyperparameters (\(\alpha_1, \alpha_2, \rho\)).

---

## 📊 Simulation Study
### Reference Parameters (\(S_{ref}\))
- \(n_A = n_B = 1000\)
- \(X \sim \mathcal{N}(m, \Sigma)\) with \(R^2 = 0.6\).
- Two scenarios:
  - **\(S_1\)**: \(\mathbb{P}_{Z^A|X^A=x} = \mathbb{P}_{Z^B|X^B=x}\) (identical conditional distributions).
  - **\(S_2\)**: \(\mathbb{P}_{Z^A|X^A=x} = \mathbb{P}_{Z^B|X^B=T(x)}\) (transformation \(T\) between domains).

- **\(S_1\)**: JDOT-wi outperforms other methods (assumption holds).
- **\(S_2\)**: JDOT-be and its variants are more robust to covariate shifts.
- Unbalanced transport (JDOT-be-un) improves performance in case of imbalance.

## Simple example

```julia

using OptimalTransportDataIntegration

params = DataParameters()
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
the covariates `X` from databases `A` and `B` and the outcomes `Y` from
database `A` and `Z` from database `B`. The package methods can be used
to obtain a recoded version of variable `Y` for database `B` and `Z` for
database `A`. The accuracy function can be used to verify the quality
of the method.

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

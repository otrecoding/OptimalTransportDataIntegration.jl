# Optimal Transport Between Datasets for Statistical Matching

---

## 📌 Overview

[`JointOTBetweenBases`](@ref) (JDOT-be) is an advanced method for
**statistical matching** that leverages **optimal transport theory**
to align joint distributions of explanatory variables and outcomes
**between** datasets. Unlike JDOT-wi, JDOT-be simultaneously optimizes
a **transport plan** and **predictive functions** to handle missing
values in a unified synthetic dataset.

---

## 🔍 Context and Motivation

### Problem Statement
In statistical matching, datasets ``A`` and ``B`` often contain distinct units but share some variables:

| Dataset  | observed | observed | unobserved |
|---------------|---------------------|---------------------|-----------------------|
| Dataset ``A`` | ``X^A`` | ``Y^A``  | ``Z^A`` |
| Dataset ``B`` | ``X^B`` | ``Z^B``  | ``Y^B`` |

**Objective**: Predict ``Z^A`` and ``Y^B`` to create a synthetic dataset where all variables are jointly available.

### Key Idea

JDOT-be **transports the joint distribution** ``(X, Y, Z)`` **between**
datasets ``A`` and ``B`` while optimizing predictive functions ``f``
and ``g`` to impute missing values. This approach is particularly
useful when datasets have **different distributions** or **covariate
shifts**.

---

## 🛠️ Methodology

### Step 1: Define the Optimal Transport Problem

JDOT-be seeks a coupling ``\gamma`` that minimizes the transportation cost between the joint distributions ``\mu^{(X^A, Y^A, Z^A)}`` and ``\mu^{(X^B, Y^B, Z^B)}``:

```math
\gamma^\star \in \arg\min_{\gamma \in \Gamma(\mu^{(X^A, Y^A, Z^A)}, \mu^{(X^B, Y^B, Z^B)})} \int c\left((x, y, z), (x', y', z')\right) d\gamma_{(x, y, z), (x', y', z')},
```

where the **cost function** ``c`` combines:
- A distance ``d(x, x')`` between explanatory variables ``X^A`` and ``X^B``.
- Loss functions ``\mathcal{L}_Y(y, y')`` and ``\mathcal{L}_Z(z, z')`` for outcomes ``Y`` and ``Z``:
```math
  c\left((x, y, z), (x', y', z')\right) = d(x, x') + \alpha_1 \mathcal{L}_Y(y, y') + \alpha_2 \mathcal{L}_Z(z, z').
```
 
``\alpha_1`` and ``\alpha_2`` are hyperparameters balancing the alignment of explanatory variables and outcomes.

---

### Step 2: Handle Unobserved Variables

Since ``Z^A`` and ``Y^B`` are unobserved, JDOT-be replaces them with predictions:
- ``Z^A`` is predicted using ``f(X^A, Y^A)``.
- ``Y^B`` is predicted using ``g(X^B, Z^B)``.

The empirical version of the problem becomes:

```math
\min_{f, g, \gamma} \sum_{i \in A, j \in B} c\left((X_i, Y_i, f(X_i, Y_i)), (X_j, g(X_j, Z_j), Z_j)\right) \gamma_{i,j},
```

subject to:

```math
\sum_{i \in A} \gamma_{i,j} = \frac{1}{n}, \quad \forall j \in B,
```

```math
\sum_{j \in B} \gamma_{i,j} = \frac{1}{n}, \quad \forall i \in A,
```

```math
\gamma_{i,j} \geq 0, \quad \forall i \in A, j \in B.
```

---

### Step 3: Solve the Optimization Problem

JDOT-be uses a **Block Coordinate Descent (BCD)** algorithm to alternate between:
1. **Updating the transport plan** ``\gamma`` for fixed ``f`` and ``g``.
2. **Optimizing the predictive functions** ``f`` and ``g`` for fixed ``\gamma``.

#### Predictive Functions

- ``f: \mathcal{X} \times \mathcal{Y} \to \mathcal{Z}`` predicts ``Z^A`` in dataset ``A``.
- ``g: \mathcal{X} \times \mathcal{Z} \to \mathcal{Y}`` predicts ``Y^B`` in dataset ``B``.

These functions are typically implemented using **neural networks** or other machine learning models.

---

### Step 4: Unbalanced Optimal Transport (Optional)
For **imbalanced datasets**, JDOT-be can use **unbalanced optimal transport** to relax marginal constraints. The unbalanced OT problem is defined as:

```math
\gamma^\star \in \arg\min_{\gamma \geq 0} \sum c\left((x, y, z), (x', y', z')\right) \gamma_{(x, y, z), (x', y', z')} + \rho_1 \text{KL}(\gamma^A \| \mu^{(X^A, Y^A, Z^A)}) + \rho_2 \text{KL}(\gamma^B \| \mu^{(X^B, Y^B, Z^B)}),
```

where ``\text{KL}`` is the **Kullback-Leibler divergence**, and ``\rho_1, \rho_2`` are regularization parameters.

---

## ✅ Advantages

- **Handles Covariate Shifts**: Effective when datasets have different distributions.
- **No CIA Assumption**: Unlike traditional methods, JDOT-be does not assume conditional independence between ``Y`` and ``Z`` given ``X``.
- **Flexibility**: Works with **mixed data types** (continuous and categorical variables).
- **Theoretical Guarantees**: Provides bounds on prediction error under specific assumptions.

---

## ⚠️ Limitations

- **Computational Cost**: Higher than JDOT-wi due to the joint optimization of ``\gamma``, ``f``, and ``g``.
- **Hyperparameter Sensitivity**: Performance depends on the choice of ``\alpha_1``, ``\alpha_2``, and ``\rho``.
- **Complexity**: Requires careful tuning for optimal results.

## 📂 Implementation

### Algorithmic Steps

1. **Input**: Datasets ``A`` and ``B`` with observed variables ``X, Y`` (for ``A``) and ``X, Z`` (for ``B``).
2. **Initialize**: Set initial predictive functions ``f_0`` and ``g_0``.
3. **Alternate Optimization**:
   - Update the transport plan ``\gamma`` for fixed ``f`` and ``g``.
   - Optimize ``f`` and ``g`` for fixed ``\gamma``.
4. **Output**: Predicted values ``\hat{Z}^A = f(X^A, Y^A)`` and ``\hat{Y}^B = g(X^B, Z^B)``.


```@autodocs
Modules = [OptimalTransportDataIntegration]
Pages = [
"joint_ot_between_bases_jdot.jl",
"joint_ot_between_bases_with_predictors.jl",
"joint_ot_between_bases_without_outcomes.jl",
"joint_ot_between_bases_discrete_ordered.jl",
"joint_ot_between_bases_da_covariables.jl",
"joint_ot_between_bases_da_outcomes.jl",
"joint_ot_between_bases_da_outcomes_with_predictors.jl",
"joint_ot_between_bases_discrete.jl"]
```

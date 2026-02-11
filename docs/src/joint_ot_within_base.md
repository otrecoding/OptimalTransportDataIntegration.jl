# Optimal transport of the joint distribution within a data source


## 📌 Overview
**JDOT-wi** (Joint Distribution Optimal Transport - Within) is a method designed to integrate distinct datasets by leveraging **optimal transport theory**. It focuses on aligning the joint distributions of **explanatory variables** and **outcomes** **within** each dataset, enabling the prediction of missing values in a unified synthetic dataset.

---

## 🔍 Context and Motivation
### Problem Statement
In statistical matching, we often encounter two datasets, \(A\) and \(B\), with the following structure:

| Dataset \(A\) | \(X^A\) (observed) | \(Y^A\) (observed) | \(Z^A\) (unobserved) |
|---------------|---------------------|---------------------|-----------------------|
| Dataset \(B\) | \(X^B\) (observed) | \(Y^B\) (unobserved) | \(Z^B\) (observed) |

**Objective**: Predict the unobserved variables \(Z^A\) and \(Y^B\) to create a synthetic dataset where all variables are jointly available.

### Key Assumption
JDOT-wi assumes that the **conditional distributions** of the outcomes given the explanatory variables are identical between datasets \(A\) and \(B\):
\[
\mathbb{P}(Y^A = y \mid X^A = x) = \mathbb{P}(Y^B = y \mid X^B = x),
\]
\[
\mathbb{P}(Z^A = z \mid X^A = x) = \mathbb{P}(Z^B = z \mid X^B = x).
\]

---

## 🛠️ Methodology

### Step 1: Estimate Joint Distributions
JDOT-wi begins by estimating the joint distributions of the observed variables within each dataset:

1. **Empirical Estimators**:
   For dataset \(A\), the joint distribution \(\mu^{(X^A, Y^A)}\) is estimated as:
   \[
   \hat{\mu}_{n, x, y}^{(X^A, Y^A)} = \frac{1}{n} \sum_{i \in A} \mathbf{1}_{\{X_i = x, Y_i = y\}}.
   \]

   Similarly, the joint distribution \(\mu^{(X^B, Z^B)}\) is estimated for dataset \(B\).

2. **Proxy for \(\mu^{(X^A, Z^A)}\)**:
   Since \(Z^A\) is unobserved, JDOT-wi uses the joint distribution from dataset \(B\) to estimate \(\mu^{(X^A, Z^A)}\):
   \[
   \hat{\mu}_{n, x, z}^{(X^A, Z^A)} =
   \begin{cases}
   \frac{\hat{\mu}_{n, x, z}^{(X^B, Z^B)} \cdot \hat{\mu}_{n, x}^{X^A}}{\hat{\mu}_{n, x}^{X^B}}, & \text{if } \hat{\mu}_{n, x}^{X^B} \neq 0, \\
   0, & \text{otherwise.}
   \end{cases}
   \]

---

### Step 2: Optimal Transport Problem
JDOT-wi formulates an **optimal transport problem** to align the joint distributions \(\mu^{(X^A, Y^A)}\) and \(\mu^{(X^A, Z^A)}\) within dataset \(A\). The goal is to find a coupling \(\gamma\) that minimizes the transportation cost:

\[
\min_{\gamma} \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} \sum_{z \in \mathcal{Z}} \hat{c}_{x, y, z} \gamma_{x, y, z},
\]

subject to the constraints:
\[
\sum_{z \in \mathcal{Z}} \gamma_{x, y, z} = \hat{\mu}_{x, y}^{(X^A, Y^A)}, \quad \forall x \in \mathcal{X}, y \in \mathcal{Y},
\]
\[
\sum_{y \in \mathcal{Y}} \gamma_{x, y, z} = \hat{\mu}_{x, z}^{(X^A, Z^A)}, \quad \forall x \in \mathcal{X}, z \in \mathcal{Z},
\]
\[
\gamma_{x, y, z} \geq 0, \quad \forall x \in \mathcal{X}, y \in \mathcal{Y}, z \in \mathcal{Z}.
\]

**Cost Function**:
The cost \(\hat{c}_{x, y, z}\) is estimated as the expected distance between explanatory variables \(X^A\) and \(X^B\) given \(Y^A = y\) and \(Z^B = z\):
\[
\hat{c}_{n, y, z} = \frac{1}{\kappa_{n, y, z}} \sum_{i \in A} \sum_{j \in B} \mathbf{1}_{\{Y_i = y, Z_j = z\}} \cdot d_X(X_i, X_j),
\]
where \(\kappa_{n, y, z} = \sum_{i \in A} \sum_{j \in B} \mathbf{1}_{\{Y_i = y, Z_j = z\}}\).

---

### Step 3: Predict Missing Values
Once the optimal transport plan \(\gamma\) is obtained, JDOT-wi uses it to estimate the conditional distribution \(\mu^{(Z^A \mid X^A, Y^A)}\):

\[
\hat{\mu}_{n, z}^{(Z^A \mid X^A = x, Y^A = y)} =
\begin{cases}
\frac{\hat{\gamma}_{n, x, y, z}}{\hat{\mu}_{n, x, y}^{(X^A, Y^A)}}, & \text{if } \hat{\mu}_{n, x, y}^{(X^A, Y^A)} \neq 0, \\
0, & \text{otherwise.}
\end{cases}
\]

A predictive function \(f: \mathcal{X} \times \mathcal{Y} \to \mathcal{Z}\) is then defined to predict \(Z^A\):
\[
f(x, y) \in \arg \min_{z' \in \mathcal{Z}} \sum_{z \in \mathcal{Z}} \mathcal{L}_Z(z, z') \hat{\gamma}_{n, x, y, z}.
\]

The same process is applied to dataset \(B\) to predict \(Y^B\) using a function \(g: \mathcal{X} \times \mathcal{Z} \to \mathcal{Y}\).

---

## ✅ Advantages
- **Theoretical Guarantees**: Provides bounds on the generalization error.
- **Flexibility**: Works well with categorical variables and can handle discretized continuous variables.
- **No CIA Assumption**: Unlike traditional statistical matching methods, JDOT-wi does not rely on the **Conditional Independence Assumption (CIA)**.

---

## ⚠️ Limitations
- **Discretization Requirement**: Continuous variables must be discretized, which may lead to a loss of information.
- **Sensitivity to Assumptions**: Performance depends on the validity of the conditional distribution assumption.
- **Computational Cost**: Solving the optimal transport problem can be computationally intensive for large datasets.

---

## 📂 Implementation
### Algorithmic Steps
1. **Input**: Datasets \(A\) and \(B\) with observed variables \(X, Y\) (for \(A\)) and \(X, Z\) (for \(B\)).
2. **Estimate Joint Distributions**: Compute \(\hat{\mu}^{(X^A, Y^A)}\) and \(\hat{\mu}^{(X^A, Z^A)}\).
3. **Solve Optimal Transport**: Find the coupling \(\gamma\) that minimizes the transportation cost.
4. **Predict Missing Values**: Use \(\gamma\) to estimate \(\hat{\mu}^{(Z^A \mid X^A, Y^A)}\) and predict \(Z^A\) and \(Y^B\).



```@autodocs
Modules = [OptimalTransportDataIntegration]
Pages   = ["instance.jl", 
           "joint_ot_within_base_continuous.jl", 
           "joint_ot_within_base_discrete.jl", 
           "average_distance_closest.jl", "solution.jl"]
```

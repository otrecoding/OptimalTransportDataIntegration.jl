# Simple Learning for Statistical Matching

---

## 📌 Overview

[`SimpleLearning`](@ref) is a baseline method for **statistical matching** that uses traditional **supervised learning** techniques to predict missing values in datasets. Unlike advanced methods such as [`JointOTWithinBase`](@ref) or [`JointOTBetweenBases`](@ref), `SimpleLearning` does not leverage **optimal transport** or joint distribution alignment. Instead, it independently trains predictive models on observed data to impute missing values.

- Train a model on dataset $A$ to predict $Z^A$ using $X^A$ and $Y^A$.
- Train a separate model on dataset $B$ to predict $Y^B$ using $X^B$ and $Z^B$.

This approach is straightforward and does not require complex assumptions or optimizations.

---

## 🛠️ Methodology

### Step 1: Train Predictive Models

1. **Predict $Z^A$ in Dataset $A$**:
   - Use observed pairs $(X_i^A, Y_i^A)$ to train a model $f: \mathcal{X} \times \mathcal{Y} \to \mathcal{Z}$.
   - Apply $f$ to predict $Z_i^A$ for all units in $A$.

2. **Predict $Y^B$ in Dataset $B$**:
   - Use observed pairs $(X_j^B, Z_j^B)$ to train a model $g: \mathcal{X} \times \mathcal{Z} \to \mathcal{Y}$.
   - Apply $g$ to predict $Y_j^B$ for all units in $B$.

### Step 2: Model Selection

Simple Learning can use a variety of **supervised learning algorithms**, such as:
- **Linear Regression** or **Logistic Regression** for simple relationships (not implemented).
- **Random Forests** or **Gradient Boosting** for non-linear relationships (not implemented).
- **Neural Networks** from [Flux.jl](http://fluxml.ai/Flux.jl/stable/) for complex patterns (used in this package).

!!! note
    The first option  with regression models are currently under development and will be available in the future.

### Step 3: Evaluation

The performance of Simple Learning is evaluated using standard metrics:
- **Accuracy** for categorical outcomes.
- **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)** for continuous outcomes.

---

## ✅ Advantages

- **Simplicity**: Easy to implement and interpret.
- **Flexibility**: Compatible with any supervised learning algorithm.
- **No Assumptions**: Does not require assumptions about joint or conditional distributions (e.g., no CIA assumption).
- **Computational Efficiency**: Faster than optimal transport-based methods.

---

## ⚠️ Limitations

- **No Distribution Alignment**: Ignores potential differences in distributions between datasets $A$ and $B$.
- **Lower Accuracy**: Typically underperforms compared to advanced methods that uses optimal transport, especially when datasets have **covariate shifts** or **different conditional distributions**.
- **Independent Predictions**: Predicts $Z^A$ and $Y^B$ separately, without leveraging joint information across datasets.

---

```@autodocs
Modules = [OptimalTransportDataIntegration]
Pages = ["simple_learning.jl", "simple_learning_with_continuous_data.jl"]
```

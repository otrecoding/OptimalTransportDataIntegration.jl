# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import sys

sys.executable
# -

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# # Methods
#
#  - **OT**  Transport of the joint distribution of covariates and outcomes within a data source ($\widehat{\mathcal{P}}_{1,n}$). The parameter of  relaxation is chosen $\alpha=0$, and regularization $\lambda=0$ 
#  - **OT-r**  Regularized transport of the joint distribution of covariates and outcomes within a data source ($\widehat{\mathcal{P}}_{1,n}$). We added a relaxation on the constraints and a regularization term. The parameter of relaxation $\alpha=0.4$, and regularization $\lambda=0.1$
#  - **OTE** Balanced transport of covariates and estimated outcomes between data sources ($\widehat{\mathcal{P}}_{3,n}$).
#       The parameters are chosen $\alpha_1=1/\max(\mathcal{L}_1)$, and $\alpha_2=\max(\mathcal{L}_2)$. 
#  - **OTE-r** Regularized unbalanced transport of covariates and estimated outcomes between data sources ($\widehat{\mathcal{P}}_{3,n}$).
#           We used 10 iterations for the BCD algorithm. We added an entropic regularization.  The parameter of   regularization is chosen  $\lambda=(0.01, 0.01)$. The parameter of relaxation is also optimized.
#
#     

# # Evaluation of the methods
#  
# To evaluate the performance of the methods, we compute the  accuracy of prediction of $Z$ in $A$ and $Y$ in $B$.
# The impact of the elements characterizing the simulations will be studied through the following scenarios.

# ## Effect of the sample size $n$.
#
# Keeping $p$, $m ^A$, $ $$m ^B$ and $a$, we investigate the impact of the $n$ choosing $n\in\left\{100,1000,10000\right\}$.

# +
csv_file = "sample_size_effect.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data = data,
            x = "nA", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left" , bbox_to_anchor=(1, 1))
plt.savefig("sample_size_effect.png")
# -

# ## Effect of the ratio $n^A/n^B$.
#
# Database A and B have different sample size $n^A$ and $n^B$ respectively.  Keeping $p$, $m ^A$, $ $$m ^B$ and $a$ and $n_A=1000$ we investigate the impact of the $n^A/n^B$ choosing $n^A/n^B\in\left\{1,2,10\right\}$.

# +
csv_file = "sample_ratio_effect.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})
data['ratio'] = data['nA'] / data['nB']

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data = data,
            x = "ratio", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left" , bbox_to_anchor=(1, 1))
plt.savefig("sample_ratio_effect.png")
# -

# ## Effect of the link between the covariates and Y and Z.
#
# Keeping $n$, $m ^A$, $ $$m ^B$ and $a$, we investigate the impact of the $p$ choosing $p\in\left\{0.2,0.4,0.6,0.8\right\}$.

# +
csv_file = "covariates_link_effect.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data = data,
            x = "p", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left" , bbox_to_anchor=(1, 1))
plt.savefig("covariates_link_effect.png")
# -

# ## Covariate shift assumption
#
# Keeping $p$ and $a$, we investigate the impact of differences in the distributions of $X^A$ and $X^B$ by considering the following four scenarios: 1: $m^A=m^B=(0,0,0)$,  $m^A=(0,0,0)$, 2: $m^B=(1,0,0)$,
#  3: $m^A=(0,0,0)$, $m^B=(1,1,0)$,
# and  4: $m^A=(0,0,0)$, $m^B=(1,2,0)$.

# +
csv_file = "covariates_shift_assumption.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})

plt.figure(figsize=(8,6))
ax = sns.boxplot(data = data,
            x = "mB", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xticks(range(4))
ax.set_xticklabels(["[0,0,0]", "[1,0,0]", "[1,1,0]" , "[1,2,0]"])
plt.savefig("covariates_shift_assumption.png")
# -

# ## Changes in Conditional distribution $Y|X$ and $Z|X$.
#
# Finally, we wish to evaluate the importance of satisfying the assumption that the distributions of $Y$ and $Z$ given $X$ are the same in the two databases. For this, we replace the quartile $t^Z$ and tertile $t^Y$ by $t^Z+ \epsilon$ and $t^Y+ \epsilon$ in database B. Keeping $p$, $m^A$ and $m^B=(0,0,0)$, $a^A= (1,1,1,1,1,1)$, we consider the following four scenarios: 1:  $\epsilon = 0$, 2: $\epsilon = 0.1$  3: $\epsilon = 0.5$,  4: $\epsilon = 1$.

# +
csv_file = "conditional_distribution.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})

plt.figure(figsize=(8,6))
ax = sns.boxplot(data = data,
            x = "epsilon", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.savefig("conditional_distribution.png")
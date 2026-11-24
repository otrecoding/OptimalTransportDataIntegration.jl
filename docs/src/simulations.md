# Evaluation of the methods


- `JDOT-wi`  Transport of the joint distribution of explanatory variables and outcomes within a data source (``\widehat{\mathcal{P}}_{1,n}`` and ``\widehat{\mathcal{P}}'_{1,n}``). The parameter of  relaxation is chosen ``\alpha=0``, and regularization ``\lambda=0`` 
- `JDOT-wi-r` Regularized transport of the joint distribution of explanatory variables and outcomes within a data source ($\widehat{\mathcal{P}}_{1,n}$ and $\widehat{\mathcal{P}}_{2,n}$). The relaxation and regularization parameters are selected by `best_parameters(:within, variable, scenario)`: for continuous data, $(\alpha,\lambda)=(1,1)$ in scenario 1 and $(2,1)$ in scenario 2; for discrete data, $(0.4,1)$ and $(0.1,0.1)$, respectively.
- `JDOT-be` Balanced transport of explanatory variables and estimated outcomes between data sources ($\widehat{\mathcal{P}}_{2,n}$). The implementation uses up to 10 iterations for the BCD algorithm. Its loss weights are $\alpha_1=1/|Ylevels|$ and $\alpha_2=1/|Zlevels|$ (four and three outcome levels, respectively).

- `JDOT-be-r-un` Regularized unbalanced transport of explanatory variables and estimated outcomes between data sources ($\widehat{\mathcal{P}}_{2,n}$). Predictors $f$ and $g$ are neural networks, not nearest-neighbor classifiers, and the BCD algorithm runs for up to 10 iterations. The entropic regularization and marginal-relaxation parameters are selected by `best_parameters(:between, variable, scenario)` and applied as `reg_m1=reg_m2=reg_m`: for continuous data, $(reg,reg_m)=(0.001,0.001)$ in scenario 1 and $(0.001,0.5)$ in scenario 2; for discrete data, $(0.01,0.01)$ and $(0.01,0.75)$, respectively.

To evaluate the performance of the methods, we compute the  accuracy of prediction of $Z$ in $A$ and $Y$ in $B$.
The impact of the elements characterizing the simulations will be studied through the following scenarios.

In both continuous and discrete data generators, scenario 1 uses the same outcome-category bin thresholds for databases A and B, while scenario 2 computes the thresholds separately from each database's outcomes. Thus, scenario 1 keeps the outcome categorization aligned across databases; scenario 2 allows it to differ with their outcome distributions.

## Effect of the sample size.

Keeping the covariate distributions and outcome effect sizes fixed, we investigate the impact of the sample size, with $n^A=n^B=n\in\left\{100,1000,5000\right\}$. See [`sample_size_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/sample_size_effect.jl).

## Effect of the ratio of data sources sizes.

Database `A` and `B` have sample sizes $n^A$ and $n^B$, respectively. Keeping the covariate distributions and outcome effect sizes fixed, with $n^A=1000$, we investigate the impact of the ratio $n^A/n^B\in\left\{1,2,5,10\right\}$. See [`sample_ratio_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/sample_ratio_effect.jl).

## Effect of the link between the covariates and Y and Z.

Keeping the sample sizes, covariate distributions, and outcome effect sizes fixed, we investigate the impact of the coefficient of determination $r^2\in\left\{0.2,0.4,0.6,0.8,1.0\right\}$. See [`covariates_link_effect.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/covariates_link_effect.jl).

## Covariate shift assumption

For continuous covariates, keeping $r^2$ and the outcome effect sizes fixed, we investigate differences in the distributions of $X^A$ and $X^B$ with the following four mean settings:

  1. ``m^A=(0,0,0)``, ``m^B=(0,0,0)``, 
  2. ``m^A=(0,0,0)``, ``m^B=(1,0,0)``,
  3. ``m^A=(0,0,0)``, ``m^B=(1,1,0)``,
  4. ``m^A=(0,0,0)``, ``m^B=(1,1,1)``.

For discrete covariates, $p^A$ is fixed at $((0.5,0.5),(1/3,1/3,1/3),(0.25,0.25,0.25,0.25))$. The four $p^B$ settings are:

  1. $p^B=p^A$.
  2. $p^B_1=(0.8,0.2)$, $p^B_2=p^A_2$, and $p^B_3=p^A_3$.
  3. $p^B_1=(0.8,0.2)$, $p^B_2=(0.6,0.2,0.2)$, and $p^B_3=p^A_3$.
  4. $p^B_1=(0.8,0.2)$, $p^B_2=(0.6,0.2,0.2)$, and $p^B_3=(0.6,0.2,0.1,0.1)$.

These continuous and discrete covariate-shift experiments are implemented in [`covariates_shift_assumption.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/covariates_shift_assumption.jl).

## Changes in Conditional distribution $Y|X$ and $Z|X$.

Finally, we wish to evaluate the importance of satisfying the
assumption that the distributions of $Y$ and $Z$ given $X$ are the
same in the two databases. The generator shifts the quartile thresholds
for $Y$ and the tertile thresholds for $Z$ in database B by adding
$\epsilon$. With the other parameters at their `DataParameters()`
defaults ($m^A=(0,0,0)$, $m^B=(1,1,0)$, and $a^A=a^B=(1,1,1)$), we
consider $\epsilon\in\{0,1,2,3,5\}$ in scenario 1. The discrete
experiment is run by default; the continuous invocation is commented
out in [`conditional_distribution.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/conditional_distribution.jl).

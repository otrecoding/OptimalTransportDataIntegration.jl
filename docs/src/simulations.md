# Evaluation of the methods


- `JDOT-wi`  Transport of the joint distribution of explanatory variables and outcomes within a data source (``\widehat{\mathcal{P}}_{1,n}`` and ``\widehat{\mathcal{P}}'_{1,n}``). The parameter of  relaxation is chosen ``\alpha=0``, and regularization ``\lambda=0`` 
- `JDOT-wi-r`  Regularized transport of the joint distribution of explanatory variables and outcomes within a data source ($\widehat{\mathcal{P}}_{1,n}$ and $\widehat{\mathcal{P}}_{2,n}$). We added a relaxation on the constraints and a regularization term. The parameter of  relaxation is chosen $\alpha=0.4$, and regularization $\lambda=0.7$ 
- `JDOT-be` Balanced transport of explanatory variables and estimated outcomes between data sources ($\widehat{\mathcal{P}}_{2,n}$).  We used 10 iterations for the BCD algorithm. The parameters are chosen $\alpha_1=1/\max(\mathcal{L}_1)$, and $\alpha_2=1/\max(\mathcal{L}_2)$. 

- `JDOT-be-r-un` Regularized unbalanced transport of explanatory variables and estimated outcomes between data sources ($\widehat{\mathcal{P}}_{2,n}$). For classifiers $f$ and $g$, we used nearest neighbor method with one neighbor.  We used 10 iterations for the BCD algorithm. We added an entropic regularization.  The parameter of regularization is chosen  $\lambda = 0.01$ and the parameter of relaxation is $m = 0.01$.

To evaluate the performance of the methods, we compute the  accuracy of prediction of $Z$ in $A$ and $Y$ in $B$.
The impact of the elements characterizing the simulations will be studied through the following scenarios.

## Effect of the sample size.

Keeping $p$, $m^A$, $m^B$ and $a$, we investigate the impact of the $n$ choosing $n\in\left\{100,1000,10000\right\}$.

## Effect of the ratio of data sources sizes.

Database `A` and `B` have different sample size $n^A$ and $n^B$ respectively.  Keeping $m^A$, $m^B$ and $a$ and $n_A=1000$ we investigate the impact of the $n^A/n^B$ choosing $n^A/n^B\in\left\{1,2,10\right\}$.

## Effect of the link between the covariates and Y and Z.

Keeping $n$, $m^A$, $m^B$ and $a$, we investigate the impact of the $p$ choosing $p\in\left\{0.2,0.4,0.6,0.8\right\}$.

## Covariate shift assumption

Keeping $p$ and $a$, we investigate the impact of differences in the distributions of $X^A$ and $X^B$ by considering the following four scenarios: 

  1. ``m^A=(0,0,0)``, ``m^B=(0,0,0)``, 
  2. ``m^A=(0,0,0)``, ``m^B=(1,0,0)``,
  3. ``m^A=(0,0,0)``, ``m^B=(1,1,0)``,
  4. ``m^A=(0,0,0)``, ``m^B=(1,2,0)``.

## Changes in Conditional distribution $Y|X$ and $Z|X$.

Finally, we wish to evaluate the importance of satisfying the
assumption that the distributions of $Y$ and $Z$ given $X$ are the
same in the two databases. For this, we replace the quartile $t^Z$
and tertile $t^Y$ by $t^Z+ \epsilon$ and $t^Y+ \epsilon$ in database
B. Keeping $p$, $m^A$ and $m^B=(0,0,0)$, $a^A= (1,1,1,1,1,1)$, we
consider the following four scenarios: ``\epsilon = (0, 0.1, 0.5, 1)``.

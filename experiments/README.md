[TOC]

# OptimalTransportDataIntegration.jl : Numerical experiments

- Scenario 1 : $Y^A$ | $X^A$ $\sim$  $Y^B$ | $X^B$
- Scenario 2 : $Y^A$ | $X^A$ $\sim$  $Y^B$ | $T(X^B)$

## Continuous data

### Scenario 1 

Dimension of the continuous covariable is 3

Reference scenario:
- n=1000 
- $n^A/n^B=1$
- $R^2=0.6$
- $m^A=(0,0,0)$ $m^B=(1,1,0)$ 
- $\epsilon = 0$

- Effect of the sample size $n=100,1000,10000$
- Effect of the ratio $n^A/n^B=1,2,10$
- Effect of the link between the explanatory variables and $Y$ and $Z$ $R^2=0.2,0.4,0.6,0.8^
- Covariate shift assumption 
$$
m^B=[(0,0,0),(1,0,0),(1,1,0),(1,1,1)] 
$$ 


- Changes in conditional distributions $Y$ | $X$ and $Z$ | $X$ $\epsilon = 0,0.1,0.5,1$

Methods

- Simple learning
- OT within
- OT within + relaxation + regularisation
- OT between (with emd)
- OT beween + relaxation + regularisation

### Scenario 2 

continuous covariables with dimension 3. Outcomes  $Y$ and $Z$ are categorical with levels equal to 1:4 and 1:3 respectively.

- Covariate shift assumption 
$$
m^B=[(0,0,0), (1,0,0), (1,1,0), (1,1,1)]
$$

Methods

- Simple learning
- OT within
- OT within + relaxation + regularization
- OT between (with emd)
 

## Discrete data

Discrete covariables of dimension 3. First dimension levels are 1:2, second dimension 1:3 and third dimension 1:4.

### Scenario 1

Parameters:

- n=1000 
- $n^A/n^B=1$ 
- $R^2=0.6$ ^
- $\epsilon = 0$
- $p^A_1=(0.5,0.5)$, $p^A_2=(1/3,1/3,1/3)$, $p^A_3=(0.25,0.25,0.25,0.25)$ 
- $p^B_1=(0.8,0.2)$, $p^B_2=p^A_2$, $p^B_3=p^A_3$ 
    
- Covariate shift assumption 
- Changes in Conditional distributions $Y$ | $X$ and $Z$ | $X$ $\epsilon = 0,0.1,0.5,1$

## Scenario 2 - discrete data

same data as scenario 1

### Covariate shift assumption

1. $p^B_1=p^A_1$, $p^B_2=p^A_2$, $p^B_3=p^A_3$
2. $p^B_1=(0.8,0.2)$, $p^B_2=p^A_2$, $p^B_3=p^A_3$ 
3. $p^B_1=(0.8,0.2)$, $p^B_2=(0.6,0.2,0.2)$, $p^B_3=p^A_3$ 
4. $p^B_1=(0.8,0.2)$, $p^B_2=(0.6,0.2,0.2)$, $p^B_3=(0.6,0.2,0.1,0.1)$ 
5. Compare  continuous covariables (dim=3)


reference scenario
scenario 1 et scenario 2

- **SL**: Simple learning.  Learn $f$ with $(X^A_i, Y^A_i)$ and $\widehat{Y}^B_j=\widehat{f}(X^B_j)$. Same for $Z^B_j$
- **OTDA(x)**: Transport $X^A$ on $X^B$. Transport $X^B_j$ in $A$, it gives $X^{B,t}_i$.  Learn $f$ with $(X^{B,t}_i,Y^{A}_i)$. $\widehat{Y}^B_j=\widehat{f}(X^B_j)$.
- **OTDA(y,z)**: Transport $X^A$ on $X^B$. Transport $Y^A_i$ in $B$, it gives $Y^{A,t}_j$.  $\widehat{Y}^B_j=Y^{A,t}_j$.
- **OTDA(y,z)**: with predictors. Transport $X^A$ on $X^B$. Transport $Y^A_i$ in $B$, it gives $Y^{A,t}_j$. Learn f with $(X^B_j,Y^{A,t}_j)$. 

$$\widehat{Y}^B_j=\widehat{f}(X^B_j)$$
$$\hat{P}^A=n\boldsymbol{\gamma}^T P^B$$
- JDOT in $A$ and $B$.
- OT between bases with $Y=f(X)$, $Z=g(X)$.
- OT between bases with $Y=f(X,Z)$, $Z=g(X,Y)$.
- OT within bases

6. Scenario 1. continous data

- $n=1000$
- $n^A/n^B=1$ 
- $R^2=0.6$ 
- $m^A=(0,0,0)$ 
- $m^B=(1,1,0)$ 
- $\epsilon = 0$

different OT beween  + regularisation avec différent paramètres de relaxation

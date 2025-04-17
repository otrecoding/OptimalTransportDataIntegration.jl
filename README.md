# OptimalTransportDataIntegration.jl

[![CI](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/otrecoding/OptimalTransportDataIntegration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/otrecoding/OptimalTransportDataIntegration.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://otrecoding.github.io/OptimalTransportDataIntegration.jl/dev)

This package implements a statistical matching strategy based on
optimal transport theory to integrate different data sources.
These data sources are related to the same target population, which
share a subset of covariates which each data source has its own
distinct subset of variables. After recoding you'll get a unique
data set in which all the variables, coming from the different
sources, are jointly available. 

This package is derived from
[OTRecod.jl](https://github.com/otrecoding/OTRecod.jl) where joint
distribution of shared and distinct variables is transported within
the data sources. Here the method also transports the distribution
of shared and distinct variables and estimates a function to predict
the missing variables.

## Installation

The package runs on julia 1.1 and above.
In a Julia session switch to `pkg>` mode to add the package:

```julia
julia>] # switch to pkg> mode
pkg> add https://github.com/otrecoding/OptimalTransportDataIntegration.jl
```

To run an example 

```julia

using OptimalTransportDataIntegration # import the package

params = DataParameters()  # Create the parameters set

rng = PDataGenerator(params)  # Create the random generator

data = generate_data( rng ) # Generate a dataset 

result = otrecod( data, JointOTWithinBase() ) # Perform the statistical matching 

println(accuracy(result))  # Print accuracies on each distinct variables and the total accuracy.

```






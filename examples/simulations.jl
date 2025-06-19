using OptimalTransportDataIntegration
using DataFrames
using JSON, CSV

include("conditional_distribution.jl")
include("covariates_shift_assumption.jl")
include("sample_ratio_effect.jl")
include("sample_size_effect.jl")

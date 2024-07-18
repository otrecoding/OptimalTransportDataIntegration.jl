module OptimalTransportDataIntegration

    using Distributions
    using DocStringExtensions
    using Parameters
    using Printf
    
    export digitize
    
    digitize(x, bins) = searchsortedlast.(Ref(bins), x)
    
    export to_categorical
    
    to_categorical(x) = sort(unique(x)) .== permutedims(x)

    include("data_parameters.jl")
    include("generate_xcat_ycat.jl")
    include("one_hot_encoder.jl")
    include("prep_data.jl")

    include("instance.jl")
    include("solution.jl")
    include("average_distance_closest.jl")

    include("pred_error.jl")
    include("otjoint.jl")
    include("ot_joint.jl")
    include("unbalanced_modality.jl")
    include("simple_learning.jl")

end

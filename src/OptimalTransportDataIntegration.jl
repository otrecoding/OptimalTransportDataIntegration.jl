module OptimalTransportDataIntegration

    using DocStringExtensions
    using Parameters
    using Distributions
    
    export digitize
    
    digitize(x, bins) = searchsortedlast.(Ref(bins), x)
    
    export to_categorical
    
    to_categorical(x) = sort(unique(x)) .== permutedims(x)

    include("data_parameters.jl")
    include("generate_xcat_ycat.jl")
    include("one_hot_encoder.jl")
    include("prep_data.jl")
    include("unbalanced_modality.jl")
    include("otjoint.jl")

end

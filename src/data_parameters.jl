export DataParameters

@with_kw struct DataParameters
    nA :: Int = 1000
    nB :: Int = 1000
    mA :: Vector{Float64} = [0, 0, 0]
    mB :: Vector{Float64} = [0, 0, 0]
    covA :: Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    covB :: Matrix{Float64} = [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]
    px1c :: Vector{Float64} = [0.5, 0.5]
    px2c :: Vector{Float64} = [0.333, 0.334, 0.333]
    px3c :: Vector{Float64} = [0.25, 0.25, 0.25, 0.25]
    p :: Float64 = 0.6
    aA :: Vector{Float64} = [1, 1, 1, 1, 1, 1]
    aB :: Vector{Float64} = [1, 1, 1, 1, 1, 1]
    eps :: Float64 = 0.0
end

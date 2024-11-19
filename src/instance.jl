using DelimitedFiles
using Distances

export Instance

"""
$(TYPEDEF)

$(TYPEDSIGNATURES)

Definition and initialization of an Instance structure

- datafile : file name
- distance : ∈ ( Cityblock, Euclidean, Hamming )
- indXA    : indexes of subjects of A with given X value
- indXB    : indexes of subjects of B with given X value
"""
struct Instance

    nA::Int64
    nB::Int64
    D::Matrix{Float64}
    Xobserv::Matrix{Int}
    Yobserv::Vector{Int}
    Zobserv::Vector{Int}
    Xlevels::Vector{Vector{Int}}
    Ylevels::Vector{Int}
    Zlevels::Vector{Int}
    indY::Dict{Int64,Vector{Int64}}
    indZ::Dict{Int64,Vector{Int64}}
    indXA::Dict{Int64,Vector{Int64}}
    indXB::Dict{Int64,Vector{Int64}}
    DA::Matrix{Float64}
    DB::Matrix{Float64}


    function Instance(base::AbstractVector, X::AbstractMatrix, 
                      Y::AbstractVector, Ylevels::AbstractVector,
                      Z::AbstractVector, Zlevels::AbstractVector,
                      distance::Distances.Metric)

        indA = findall(base .== 1)
        indB = findall(base .== 2)

        Xobserv = vcat(X[indA, :], X[indB, :])
        Yobserv = vcat(Y[indA], Y[indB])
        Zobserv = vcat(Z[indA], Z[indB])

        nA = length(indA)
        nB = length(indB)

        # list the distinct modalities in A and B
        indY = Dict((m, findall(Y[indA] .== m)) for m in Ylevels)
        indZ = Dict((m, findall(Z[indB] .== m)) for m in Zlevels)

        # compute the distance between pairs of individuals in different bases
        # devectorize all the computations to go about twice faster only compute norm 1 here
        a = X[indA, :]'
        b = X[indB, :]'

        D = pairwise(distance, a, b, dims = 2)
        DA = pairwise(distance, a, a, dims = 2)
        DB = pairwise(distance, b, b, dims = 2)

        # Compute the indexes of individuals with same covariates
        indXA = Dict{Int64,Array{Int64}}()
        indXB = Dict{Int64,Array{Int64}}()
        Xlevels = sort(unique(eachrow(X)))

        nbX = 0
        # aggregate both bases
        for x in Xlevels
            nbX = nbX + 1
            distA = vec(pairwise(distance, x[:,:], a, dims = 2))
            distB = vec(pairwise(distance, x[:,:], b, dims = 2))
            indXA[nbX] = findall(distA .< 0.1)
            indXB[nbX] = findall(distB .< 0.1)
        end

        new(
            nA,
            nB,
            D,
            Xobserv,
            Yobserv,
            Zobserv,
            Xlevels,
            Ylevels,
            Zlevels,
            indY,
            indZ,
            indXA,
            indXB,
            DA,
            DB,
        )
    end

end

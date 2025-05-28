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

    nA::Int
    nB::Int
    D::Matrix{Float64}
    Xobserv::Matrix{Int}
    Yobserv::Vector{Int}
    Zobserv::Vector{Int}
    Xlevels::Vector{Vector{Int}}
    Ylevels::Vector{Int}
    Zlevels::Vector{Int}
    indY::Vector{Vector{Int}}
    indZ::Vector{Vector{Int}}
    indXA::Vector{Vector{Int}}
    indXB::Vector{Vector{Int}}


    function Instance(
            base::AbstractVector,
            X::AbstractMatrix,
            Y::AbstractVector,
            Ylevels::AbstractVector,
            Z::AbstractVector,
            Zlevels::AbstractVector,
            distance::Distances.Metric,
        )

        indA = findall(base .== 1)
        indB = findall(base .== 2)

        Xobserv = vcat(X[indA, :], X[indB, :])
        Yobserv = vcat(Y[indA], Y[indB])
        Zobserv = vcat(Z[indA], Z[indB])

        nA = length(indA)
        nB = length(indB)

        # list the distinct modalities in A and B
        indY = [findall(Y[indA] .== m) for m in Ylevels]
        indZ = [findall(Z[indB] .== m) for m in Zlevels]

        # compute the distance between pairs of individuals in different bases
        # devectorize all the computations to go about twice faster only compute norm 1 here
        a = X[indA, :]'
        b = X[indB, :]'

        D = pairwise(distance, a, b, dims = 2)

        # Compute the indexes of individuals with same covariates
        indXA = Vector{Int64}[]
        indXB = Vector{Int64}[]
        Xlevels = sort(unique(eachrow(X)))

        # aggregate both bases
        for x in Xlevels
            distA = vec(pairwise(distance, x[:, :], a, dims = 2))
            distB = vec(pairwise(distance, x[:, :], b, dims = 2))
            push!(indXA, findall(distA .< 0.1))
            push!(indXB, findall(distB .< 0.1))
        end

        return new(
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
        )
    end

end

function Instance(
        base::AbstractVector,
        X::AbstractMatrix,
        Y::AbstractVector,
        Z::AbstractVector,
        distance::Distances.Metric,
    )
    Ylevels = sort(unique(Y))
    Zlevels = sort(unique(Z))

    return Instance(base, X, Y, Ylevels, Z, Zlevels, distance)

end

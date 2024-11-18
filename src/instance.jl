using DelimitedFiles
using Distances

export Instance

"""
$(TYPEDEF)

$(TYPEDSIGNATURES)

Definition and initialization of an Instance structure

- datafile : file name
- norme    : ( 1 : Cityblock, 2 : Euclidean, 3 : Hamming )
- indXA    : indexes of subjects of A with given X value
- indXB    : indexes of subjects of B with given X value
"""
struct Instance

    nA::Int64
    nB::Int64
    D::Matrix{Float64}
    Xval::Matrix{Float64}
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

        nA = length(indA)
        nB = length(indB)

        # list the distinct modalities in A and B
        indY = Dict((m, findall(Y[indA] .== m)) for m in Ylevels)
        indZ = Dict((m, findall(Z[indB] .== m)) for m in Zlevels)

        # compute the distance between pairs of individuals in different bases
        # devectorize all the computations to go about twice faster
        # only compute norm 1 here
        a = X[indA, :]'
        b = X[indB, :]'

        D = pairwise(distance, a, b, dims = 2)
        DA = pairwise(distance, a, a, dims = 2)
        DB = pairwise(distance, b, b, dims = 2)

        # Compute the indexes of individuals with same covariates
        A = 1:nA
        B = 1:nB
        nbX = 0
        indXA = Dict{Int64,Array{Int64}}()
        indXB = Dict{Int64,Array{Int64}}()
        Xval = collect(stack(sort(unique(eachrow(X))))')

        # aggregate both bases
        for i = axes(Xval, 1)
            nbX = nbX + 1
            x = view(Xval,i, :)
            distA = vec(pairwise(distance, x[:,:], X[A, :]', dims = 2))
            distB = vec(pairwise(distance, x[:,:], X[B.+nA, :]', dims = 2))
            indXA[nbX] = findall(distA .< 0.1)
            indXB[nbX] = findall(distB .< 0.1)
        end

        new(
            nA,
            nB,
            D,
            Xval,
            indY,
            indZ,
            indXA,
            indXB,
            DA,
            DB,
        )
    end

end

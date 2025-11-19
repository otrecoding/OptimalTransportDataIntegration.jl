export Solution

"""
$(TYPEDEF)

$(TYPEDSIGNATURES)

- tsolve       : solution time
- jointYZA     : joint distribution of Y and Z in A
- jointYZB     : joint distribution of Y and Z in B
- estimatorZA  : estimator of probability of Z for individuals in base A
- estimatorYB  : estimator of probability of Y for individuals in base B

"""
mutable struct Solution

    jointYZA::Array{Float64, 2}
    jointYZB::Array{Float64, 2}
    estimatorZA::Array{Float64, 3}
    estimatorYB::Array{Float64, 3}
    errorpredZA::Float64
    errorpredYB::Float64
    errorpredavg::Float64
    errordistribZA::Float64
    errordistribYB::Float64
    errordistribavg::Float64

    Solution(jointYZA, jointYZB) = new(jointYZA, jointYZB)

    Solution(jointYZA, jointYZB, estimatorZA, estimatorYB) =
        new(jointYZA, jointYZB, estimatorZA, estimatorYB)
end


import Base.show

function Base.show(io::IO, sol::Solution)

    println(io, " errorpredZA     : $(sol.errorpredZA)")
    println(io, " errorpredYB     : $(sol.errorpredYB)")
    println(io, " errorpredavg    : $(sol.errorpredavg)")
    println(io, " errordistribZA  : $(sol.errordistribZA)")
    println(io, " errordistribYB  : $(sol.errordistribYB)")
    println(io, " errordistribavg : $(sol.errordistribavg)")

end

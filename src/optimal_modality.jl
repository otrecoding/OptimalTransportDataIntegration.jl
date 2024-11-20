"""
    optimal_modality(value, loss, weight)

- `values`: vector of possible values
- `weight`: vector of weights 
- `loss`: matrix of size len(weight) * len(values)
- returns an argmin over value in Values of the scalar product <Loss[value,],Weight> 
"""
function optimal_modality(value, loss, weight)

    cost = Float64[]
    for j in eachindex(value)
        s = 0
        for i in eachindex(weight)
            s += loss[i, j] * weight[i]
        end
        push!(cost, s)
    end

    return values[argmin(cost)]

end

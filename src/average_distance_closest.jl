"""
$(SIGNATURES)

Compute the cost between pairs of outcomes as the average distance between
covariations of individuals with these outcomes, but considering only the
percent closest neighbors
"""
function average_distance_to_closest(inst::Instance, percent_closest::Float64)

    Y = inst.Ylevels
    Z = inst.Zlevels
    indY = inst.indY
    indZ = inst.indZ

    Davg = zeros(Float64, (length(Y), length(Z)))

    for y in Y, i in indY[y], z in Z

        nbclose = round(Int, percent_closest * length(indZ[z]))
        if nbclose > 0 
            distance = [inst.D[i, j] for j in indZ[z]]
            p = partialsortperm(distance, 1:nbclose)
            Davg[y, z] += sum(distance[p]) / nbclose / length(indY[y]) / 2.0
        end

    end

    for z in Z, j in indZ[z], y in Y

        nbclose = round(Int, percent_closest * length(indY[y]))
        if nbclose > 0 
            distance = [inst.D[i, j] for i in indY[y]]
            p = partialsortperm(distance, 1:nbclose)
            Davg[y, z] += sum(distance[p]) / nbclose / length(indZ[z]) / 2.0
        end

    end

    Davg

end

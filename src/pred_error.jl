export compute_pred_error!

"""
$(SIGNATURES)

Compute prediction errors in a solution
"""
function compute_pred_error!(
        estimatorYB,
        estimatorZA,
        inst::Instance,
        proba_disp::Bool = false,
        mis_disp::Bool = false,
    )

    A = 1:inst.nA
    B = 1:inst.nB
    Y = inst.Ylevels
    Z = inst.Zlevels
    indXA = inst.indXA
    indXB = inst.indXB
    nbX = length(indXA)

    # Count the number of mistakes in the transport
    #deduce the individual distributions of probability for each individual from the distributions
    probaZindivA = zeros(Float64, (inst.nA, length(Z)))
    probaYindivB = zeros(Float64, (inst.nB, length(Y)))
    for x in 1:nbX
        for i in indXA[x]
            probaZindivA[i, :] .= estimatorZA[x, inst.Yobserv[i], :]
        end
        for i in indXB[x]
            probaYindivB[i, :] .= estimatorYB[x, :, inst.Zobserv[i + inst.nA]]
        end
    end

    # Transport the modality that maximizes frequency
    predZA = [findmax([probaZindivA[i, z] for z in Z])[2] for i in A]
    predYB = [findmax([probaYindivB[j, y] for y in Y])[2] for j in B]

    return predYB, predZA

end

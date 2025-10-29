using DelimitedFiles
using OptimalTransportDataIntegration

function main(nsimulations::Int)

    outfile = "best_reg_m_values_with_different_mb.csv"
    reg_m_values = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    header = ["id", "pB", "reg_m", "estyb", "estza", "accuracy", "scenario"]

    pA = [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]
    pB = [[[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]]
    push!(pB, [[0.8, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]])
    push!(pB, [[0.8, 0.2], [0.6, 0.2, 0.2], [0.25, 0.25, 0.25, 0.25]])
    push!(pB, [[0.8, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.1, 0.1]])

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for shift in pB, scenario in 1:2

            params = DataParameters(pB = shift)
            rng = DiscreteDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations
                data = generate(rng)
                for reg_m in reg_m_values
                    result = otrecod(
                        data,
                        JointOTBetweenBases(reg_m1 = reg_m, reg_m2 = reg_m),
                    )
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(shift) reg_m estyb estza est scenario])
                end
            end

        end

    end

end

nsimulations = 100

@time main(nsimulations)

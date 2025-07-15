using DelimitedFiles
using OptimalTransportDataIntegration

function main(nsimulations::Int)

    outfile = "best_reg_m_values_with_different_mb.csv"
    mb_values = [[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 2, 1]]
    reg_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    reg_m_values = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    header = ["id", "mB", "reg", "reg_m", "estyb", "estza", "accuracy", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in mb_values, scenario = 1:2

            params = DataParameters(mB = mB)
            rng = DataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                for reg in reg_values, reg_m in reg_m_values

                    result = otrecod(
                        data,
                        JointOTBetweenBases(reg = reg, reg_m = reg_m),
                    )
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) reg reg_m1 reg_m2 estyb estza est])

                end

            end

        end

    end

end

nsimulations = 1000

@time main(nsimulations)

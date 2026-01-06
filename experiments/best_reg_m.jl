using DelimitedFiles
using OptimalTransportDataIntegration

function main(nsimulations::Int)

    outfile = "best_reg_m_values_with_different_mb.csv"
    reg_m_values = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    header = ["id", "mB", "reg_m", "estyb", "estza", "accuracy", "scenario"]
    shift = ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1])
    nA = 500
    nB = 500

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        scenario = 1
        for mB in shift

            params = DataParameters(nA = nA, nB = nB, mB = mB)
            rng = ContinuousDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations
                data = generate(rng)
                for reg_m in reg_m_values
                    result = otrecod(
                        data,
                        JointOTBetweenBasesWithPredictors(reg = 0.001, reg_m1 = reg_m, reg_m2 = reg_m),
                    )
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) reg_m estyb estza est scenario])
                end
            end

        end

    end

end

nsimulations = 100

@time main(nsimulations)

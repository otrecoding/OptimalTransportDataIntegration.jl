using OptimalTransportDataIntegration
import OptimalTransportDataIntegration: AbstractMethod
using DelimitedFiles

function reference_methods_continuous(start, stop)

    params = DataParameters()

    outfile = "reference_methods_continuous.csv"
    header = ["id", "est_yb", "est_za", "est", "method", "scenario"]

    return open(outfile, "w") do io

        seekstart(io)
        writedlm(io, hcat(header...))

        for scenario in 1:2

            rng = ContinuousDataGenerator(params, scenario = scenario)

            methods = Dict{String, AbstractMethod}(
                "sl" => SimpleLearning(),
                "wi-r" => JointOTWithinBase(),
                "be" => JointOTBetweenBasesWithPredictors(reg = 0.0),
                "be-x" => JointOTBetweenBasesWithoutOutcomes(reg = 0.0),
                "jdot" => JointOTBetweenBasesJDOT(reg = 0.0),
                "otda-x" => JointOTDABetweenBasesCovariables(reg = 0.0),
                "otda-yz" => JointOTDABetweenBasesOutcomes(reg = 0.0),
                "otda-yz-f" => JointOTDABetweenBasesOutcomesWithPredictors(reg = 0.0)
            )

            for i in start:stop

                data = generate(rng)

                for (name, method) in methods
                    result = otrecod(data, method)
                    est_yb, est_za, est = accuracy(result)
                    writedlm(io, [i est_yb est_za est name scenario])
                end

            end


        end

    end

end

reference_methods_continuous(1, 100)

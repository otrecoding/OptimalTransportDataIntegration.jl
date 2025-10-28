using OptimalTransportDataIntegration
import OptimalTransportDataIntegration: AbstractMethod
using DelimitedFiles

function reference_methods_continuous(start, stop)

    params = DataParameters()
    scenario = 2

    outfile = "reference_methods_continuous_scenario2.csv"
    header = ["id", "est_yb", "est_za", "est", "method", "scenario"]

    return open(outfile, "w") do io

        seekstart(io)
        writedlm(io, hcat(header...))

        rng = ContinuousDataGenerator(params, scenario = scenario)

        methods = Dict{String, AbstractMethod}("sl" => SimpleLearning(),
        "wi" => JointOTWithinBase(),
        "be-with-predictors" => JointOTBetweenBasesWithPredictors(),
        "be-without-outcomes" => JointOTBetweenBasesWithoutOutcomes(),
        "jdot" => JointOTBetweenBasesJDOT(),
        "otda-x" => JointOTDABetweenBasesCovariables(),
        "otda-yz" => JointOTDABetweenBasesOutcomes())

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

reference_methods_continuous(1, 100)


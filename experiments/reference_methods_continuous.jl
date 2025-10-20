using OptimalTransportDataIntegration
import OptimalTransportDataIntegration: AbstractMethod
using DelimitedFiles

function reference_methods_continuous(start, stop)

    alpha = collect(0:0.1:2)
    lambda = collect(0:0.1:1)
    reg = [0.001, 0.01, 0.1, 1.0]
    reg_m = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
    params = DataParameters()
    scenario = 1
    discrete = false

    outfile = "reference_methods_continuous_scenario1.csv"
    header = ["id", "est_yb", "est_za", "est", "method", "scenario", "discrete"]

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
                result = otrecod(data, JointOTWithinBase())
                est_yb, est_za, est = accuracy(result)
                writedlm(io, [i est_yb est_za est name scenario discrete])
            end

        end
        

    end

end

reference_methods_continuous(1, 100)

#import Statistics: mean
#data = CSV.read("best_parameters.csv", DataFrame)
#sort(
#    combine(groupby(data, ["alpha", "lambda"]), :estimation => mean),
#    order(:estimation_mean, rev = true),
#)
# equivalent with pandas
# import pandas as pd
# data = pd.read_csv("results_otjoint.csv", sep="\t")
# data.groupby(["alpha", "lambda"]).estimation.mean().sort_values(ascending=False)

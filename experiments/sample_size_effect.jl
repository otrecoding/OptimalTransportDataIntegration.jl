using DelimitedFiles
using OptimalTransportDataIntegration

# +
function sample_size_effect_continuous(all_params, nsimulations)

    outfile = "sample_size_effect_continuous.csv"
    header = ["id", "nA", "nB", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for params in all_params, scenario in (1, 2)

            rng = ContinuousDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi" scenario])

                alpha, lambda = best_parameters(:within, :continuous, scenario)
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi-r" scenario])

                result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be" scenario])

                reg, reg_m = best_parameters(:between, :continuous, scenario)
                result = otrecod(
                    data, JointOTBetweenBasesWithPredictors(
                        reg = reg,
                        reg_m1 = reg_m, reg_m2 = reg_m
                    )
                )
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be-un-r" scenario])

                println("SL Simple Learning")
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "sl"  scenario])


            end

        end

    end

end

function sample_size_effect_discrete(all_params, nsimulations)

    outfile = "sample_size_effect_discrete.csv"
    header = ["id", "nA", "nB", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for params in all_params, scenario in (1, 2)

            rng = DiscreteDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                #within non-regularized
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi" scenario])

                #within regularized
                alpha, lambda = best_parameters(:within, :discrete, scenario)
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi-r" scenario])

                #between with predictors
                result = otrecod(data, JointOTBetweenBases(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be" scenario])

                #between with predictors
                reg, reg_m = best_parameters(:between, :discrete, scenario)
                result = otrecod(data, JointOTBetweenBases(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be-un-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "sl"  scenario])


            end

        end

    end

end

all_params = [
    DataParameters(nA = 100, nB = 100),
    DataParameters(nA = 1000, nB = 1000),
    DataParameters(nA = 5000, nB = 5000),
]

nsimulations = 100

@time sample_size_effect_continuous(all_params, nsimulations)
@time sample_size_effect_discrete(all_params, nsimulations)

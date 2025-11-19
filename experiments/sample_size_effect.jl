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

                println("within non-regularized")
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi" scenario])

                println("within regularized")
                result = otrecod(data, JointOTWithinBase(alpha = 0.9, lambda = 0.1))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi-r" scenario])

                println("between with predictors - emd")
                result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be" scenario])

                println("between with predictors - unbalanced")
                result = otrecod(
                    data, JointOTBetweenBasesWithPredictors(
                        reg = 0.001,
                        reg_m1 = 0.01, reg_m2 = 0.01
                    )
                )
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be-r" scenario])

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
                alpha, lambda = 0.4, 0.1
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "wi-r" scenario])

                #between with predictors
                result = otrecod(data, JointOTBetweenBases(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be" scenario])

                #between with predictors
                result = otrecod(data, JointOTBetweenBases(reg = 0.001, reg_m1 = 0.25, reg_m2 = 0.25))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i params.nA params.nB estyb estza est "be-r" scenario])

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

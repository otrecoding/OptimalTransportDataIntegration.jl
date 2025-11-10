using DelimitedFiles
using OptimalTransportDataIntegration

function conditional_distribution_continuous(nsimulations::Int, epsilons)

    outfile = "conditional_distribution_continuous.csv"
    header = ["id", "epsilon", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        params = DataParameters()

        scenario = 1

        rng = ContinuousDataGenerator(params, scenario = scenario)

        for eps in epsilons

            for i in 1:nsimulations

                data = generate(rng, eps = eps)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "wi" scenario])

                #OT-r Regularized Transport
                alpha, lambda = 0.9, 0.8
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "wi-r" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "be" scenario])

                #OTE Regularized unbalanced transport
                result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.001, reg_m1 = 0.01, reg_m2 = 0.01))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "be-un-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "sl" scenario])

            end

        end

    end

end

function conditional_distribution_discrete(nsimulations::Int, epsilons)

    outfile = "conditional_distribution_discrete.csv"
    header = ["id", "epsilon", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        params = DataParameters()

        scenario = 1

        rng = DiscreteDataGenerator(params, scenario = scenario)

        for eps in epsilons

            for i in 1:nsimulations

                data = generate(rng, eps = eps)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "wi" scenario])

                #OT-r Regularized Transport
                alpha, lambda = 0.4, 0.1
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "wi-r" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "be" scenario])

                #OTE Regularized unbalanced transport
                result = otrecod(data, JointOTBetweenBases(reg = 0.001, reg_m1 = 0.25, reg_m2 = 0.25))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "be-un-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i eps estyb estza est "sl" scenario])

            end

        end

    end

end

nsimulations = 100
epsilons = (0.0, 0.1, 0.5, 1.0)

@time conditional_distribution_continuous(nsimulations, epsilons)
@time conditional_distribution_discrete(nsimulations, epsilons)

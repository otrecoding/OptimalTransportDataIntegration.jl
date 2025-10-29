using DelimitedFiles
using OptimalTransportDataIntegration

function covariates_link_effect_continuous(nsimulations::Int, r2values)

    outfile = "covariates_link_effect_continuous.csv"
    header = ["id", "r2", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for r2 in r2values, scenario in (1, 2)

            params = DataParameters(r2 = r2)

            rng = ContinuousDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "wi" scenario])

                #OT-r Regularized Transport
                result = otrecod(data, JointOTWithinBase())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "wi-r" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "be-un" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBasesWithPredictors())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "be-un-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "sl" scenario])

            end

        end

    end

end

function covariates_link_effect_discrete(nsimulations::Int, r2values)

    outfile = "covariates_link_effect_discrete.csv"
    header = ["id", "r2", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for r2 in r2values, scenario in (1, 2)

            params = DataParameters(r2 = r2)

            rng = DiscreteDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                #OT Transport of the joint distribution of covariates and outcomes.
                alpha, lambda = 0.0, 0.0
                result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "wi" scenario])

                #OT-r Regularized Transport
                result = otrecod(data, JointOTWithinBase())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "wi-r" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg = 0.0))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "be-un" scenario])

                #OTE Balanced transport of covariates and estimated outcomes
                result = otrecod(data, JointOTBetweenBases(reg = 0.001, reg_m1 = 0.25, reg_m2 = 0.25))
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "be-un-r" scenario])

                #SL Simple Learning
                result = otrecod(data, SimpleLearning())
                estyb, estza, est = accuracy(result)
                writedlm(io, [i r2 estyb estza est "sl" scenario])

            end

        end

    end

end

nsimulations = 100

@time covariates_link_effect_continuous(nsimulations, (0.2, 0.4, 0.6, 0.8, 1.0))
@time covariates_link_effect_discrete(nsimulations, (0.2, 0.4, 0.6, 0.8, 1.0))

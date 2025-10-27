using DelimitedFiles
using OptimalTransportDataIntegration

function covariates_shift_assumption_continuous(nsimulations::Int, shift)

    outfile = "covariates_shift_assumption_continuous.csv"
    header = ["id", "mB", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in shift

            params = DataParameters(mB = mB)

            for scenario in 1:2

                rng = ContinuousDataGenerator(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)

                    alpha, lambda = 0.0, 0.0
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi" scenario])

                    result = otrecod(data, JointOTWithinBase())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi-r" scenario])

                    result = otrecod(data, JointOTBetweenBasesWithPredictors(reg=0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be" scenario])

                    result = otrecod(data, JointOTBetweenBasesWithPredictors())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be-r" scenario])

                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "sl" scenario])

                end

            end

        end

    end

end

function covariates_shift_assumption_discrete(nsimulations::Int, shift)

    outfile = "covariates_shift_assumption_discrete.csv"
    header = ["id", "mB", "estyb", "estza", "est", "method", "scenario"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in shift

            params = DataParameters(mB = mB)

            for scenario in 1:2

                rng = DiscreteDataGenerator(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)

                    alpha, lambda = 0.0, 0.0
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi" scenario])

                    result = otrecod(data, JointOTWithinBase())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi-r" scenario])

                    result = otrecod(data, JointOTBetweenBases(reg=0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be" scenario])

                    result = otrecod(data, JointOTBetweenBases(reg=0.001, reg_m1=0.25, reg_m2=0.25))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be-r" scenario])

                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "sl" scenario])

                end

            end

        end

    end

end

nsimulations = 100
shift = ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1])
@time covariates_shift_assumption_continuous(nsimulations, shift)
@time covariates_shift_assumption_discrete(nsimulations, shift)

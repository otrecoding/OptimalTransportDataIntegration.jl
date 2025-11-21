using DelimitedFiles
using OptimalTransportDataIntegration

function covariates_shift_assumption_continuous(nsimulations::Int)

    outfile = "covariates_shift_assumption_continuous.csv"
    header = ["id", "mB", "estyb", "estza", "est", "method", "scenario"]
    shift = ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1])

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

                    alpha, lambda = best_parameters(:within, :continuous, scenario)
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi-r" scenario])

                    result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be" scenario])

                    reg, reg_m = best_parameters(:between, :continuous, scenario)
                    result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be-un-r" scenario])

                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "sl" scenario])

                end

            end

        end

    end

end

function covariates_shift_assumption_discrete(nsimulations::Int)

    outfile = "covariates_shift_assumption_discrete.csv"
    header = ["id", "pB", "estyb", "estza", "est", "method", "scenario"]

    pA = [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]
    pB = [[[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]]]
    push!(pB, [[0.8, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.25, 0.25, 0.25, 0.25]])
    push!(pB, [[0.8, 0.2], [0.6, 0.2, 0.2], [0.25, 0.25, 0.25, 0.25]])
    push!(pB, [[0.8, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.1, 0.1]])

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for shift in pB

            params = DataParameters(pB = shift)

            for scenario in 1:2

                rng = DiscreteDataGenerator(params, scenario = scenario)

                for i in 1:nsimulations

                    data = generate(rng)

                    alpha, lambda = 0.0, 0.0
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(shift) estyb estza est "wi" scenario])

                    alpha, lambda = best_parameters(:within, :discrete, scenario)
                    result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(shift) estyb estza est "wi-r" scenario])

                    result = otrecod(data, JointOTBetweenBases(reg = 0.0))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(shift) estyb estza est "be" scenario])

                    reg, reg_m = best_parameters(:between, :discrete, scenario)
                    result = otrecod(data, JointOTBetweenBases(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(shift) estyb estza est "be-un-r" scenario])

                    result = otrecod(data, SimpleLearning())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(shift) estyb estza est "sl" scenario])

                end

            end

        end

    end

end

nsimulations = 100
@time covariates_shift_assumption_continuous(nsimulations)
@time covariates_shift_assumption_discrete(nsimulations)

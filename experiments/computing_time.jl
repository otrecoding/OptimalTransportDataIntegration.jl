using OptimalTransportDataIntegration
using DelimitedFiles
using TimerOutputs

const to = TimerOutput()

function computing_time(nsimulations)

    params = DataParameters()
    outfile = "computing_time.txt"

    return open(outfile, "w") do io

        seekstart(io)

        for scenario in 1:2

            rng = ContinuousDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                alpha, lambda = 0.0, 0.0
                @timeit to "c-wi" result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))

                alpha, lambda = best_parameters(:within, :continuous, scenario)
                @timeit to "c-wi-r" result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))

                @timeit to "c-be" result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = 0.0))

                reg, reg_m = best_parameters(:between, :continuous, scenario)
                @timeit to "c-be-r" result = otrecod(data, JointOTBetweenBasesWithPredictors(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))

                @timeit to "c-sl" result = otrecod(data, SimpleLearning())

            end

            rng = DiscreteDataGenerator(params, scenario = scenario)

            for i in 1:nsimulations

                data = generate(rng)

                alpha, lambda = 0.0, 0.0
                @timeit to "d-wi" result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))

                alpha, lambda = best_parameters(:within, :discrete, scenario)
                @timeit to "d-wi-r" result = otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda))

                @timeit to "d-be" result = otrecod(data, JointOTBetweenBasesDiscreteOrdered(reg = 0.0))

                reg, reg_m = best_parameters(:between, :continuous, scenario)
                @timeit to "d-be-r" result = otrecod(data, JointOTBetweenBasesDiscreteOrdered(reg = reg, reg_m1 = reg_m, reg_m2 = reg_m))

                @timeit to "d-sl" result = otrecod(data, SimpleLearning())

            end

        end

        write(io, string(to))

    end

end

nsimulations = 1

computing_time(nsimulations)

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,jl
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Julia 1.11.1
#     language: julia
#     name: julia-1.11
# ---

using DelimitedFiles
using OptimalTransportDataIntegration

# +
function compare_scenario(nsimulations::Int)

    outfile = "scenario_comparison.csv"
    header = ["id", "mB", "estyb", "estza", "estimation", "method", "scenario", "discrete"]

    return open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for mB in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0])

            for scenario in (1, 2), discrete in (true, false)

                params = DataParameters(mB = mB)
                rng = DataGenerator(params, scenario = scenario, discrete = discrete)

                for i in 1:nsimulations

                    data = generate(rng)

                    #OT-r Regularized Transport
                    result = otrecod(data, JointOTWithinBase())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "wi-r" scenario discrete])

                    #OTE Regularized unbalanced transport
                    result = otrecod(data, JointOTBetweenBases())
                    estyb, estza, est = accuracy(result)
                    writedlm(io, [i repr(mB) estyb estza est "be-un-r" scenario discrete])

                end

            end

        end

    end

end

nsimulations = 100

@time compare_scenario(nsimulations)

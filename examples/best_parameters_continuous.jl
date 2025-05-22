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
# ---

using OptimalTransportDataIntegration
using DelimitedFiles

function otjoint(start, stop)

    alpha = collect(0:0.1:2)
    lambda = collect(0:0.1:1)
    estimations = Float32[]

    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0])

    rng = DataGenerator(params, scenario = 1, discrete = false)

    outfile = "results_continuous.csv"
    header = ["id", "alpha", "lambda", "est_yb", "est_za",  "est", "method"]

    open(outfile, "a") do io

        for i = start:stop

            if i == 1
                seekstart(io)
                writedlm(io, hcat(header...))
            end

            data = generate(rng)

            for m in alpha, λ in lambda

                result = otrecod(data, JointOTWithinBase(alpha = m, lambda = λ))
                est_yb, est_za, est = accuracy(result)
                writedlm(io, [i m λ est_yb est_za est "within"])

            end

            m, λ = 0.0, 0.0
            result = otrecod(data, JointOTBetweenBases())
            est_yb, est_za, est = accuracy(result)
            writedlm(io, [i m λ est_yb est_za est "between"])

            m, λ = 0.0, 0.0
            result = otrecod(data, SimpleLearning())
            est_yb, est_za, est = accuracy(result)
            writedlm(io, [i m λ est_yb est_za est "learning"])

        end

    end

end

otjoint(1, 1000)

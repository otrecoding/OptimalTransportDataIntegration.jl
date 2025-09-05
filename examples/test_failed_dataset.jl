using OptimalTransportDataIntegration
using DataFrames
using CSV

function otjoint(csv_file)

    alpha = collect(0:0.1:2)
    lambda = collect(0:0.1:1)
    estimations = Float32[]

    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], p = 0.2)
    rng = DiscreteDataGenerator(params)

    data = CSV.read(joinpath("datasets", csv_file), DataFrame)
    @show csv_file
    for m in alpha, λ in lambda

        yb, za = otrecod(data, JointOTWithinBase(alpha = m, lambda = λ))
        est_yb, est_za, est = accuracy(data, yb, za)

        println(["otjoint" m λ est])

    end

    return
end

csv_file = "dataset0533.csv"
otjoint(csv_file)

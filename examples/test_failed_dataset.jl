using OptimalTransportDataIntegration
using DataFrames
using CSV

function otjoint(csv_file)

    alpha = collect(0:0.1:2)
    lambda = collect(0:0.1:1)
    estimations = Float32[]

    params = DataParameters(nA = 1000, nB = 1000, mB = [1, 0, 0], eps = 0.0, p = 0.2)

    data = CSV.read(joinpath("datasets", csv_file), DataFrame)
    @show csv_file
    for m in alpha, λ in lambda

        est = otrecod(data, JointOTWithinBase(alpha = m, lambda = λ))

        println(["otjoint" m λ est])

    end

end

csv_file = "dataset0533.csv"
otjoint(csv_file)

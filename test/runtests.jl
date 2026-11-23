using Aqua
using CSV
using DataFrames
using Documenter
using Distances
using OptimalTransportDataIntegration
using Random
using Test

@testset "Within-base zero cases" begin
    ot = OptimalTransportDataIntegration
    base = [1, 1, 2, 2]
    inst = Instance(base, reshape([1, 2, 1, 2], :, 1),
        [1, 1, 1, 1], 1:4, [1, 1, 1, 1], 1:3, Euclidean())
    costs = ot.average_distance_to_closest(inst, 1.0)
    @test costs[1, 1] ≈ 0.5
    @test all(iszero, costs[2:end, :])
    @test all(iszero, costs[:, 2:end])
    za, yb = ot.ot_joint(inst, 0.0, 0.0, 1.0)
    @test all(isfinite, za) && all(isfinite, yb)
    @test all(za[:, 2:end, 1] .== 1.0)
    @test all(iszero, za[:, 2:end, 2:end])
    @test all(iszero, yb[:, :, 2:end])

    missing_in_B = Instance(
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        reshape([1, 1, 2, 2, 3, 3, 1, 1, 1, 2], :, 1),
        fill(1, 10), 1:4, [1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 1:3, Euclidean()
    )
    za, yb = ot.ot_joint(missing_in_B, 0.0, 0.0, 1.0)
    @test za[1, 1, :] ≈ [1.0, 0.0, 0.0]
    @test za[2, 1, :] ≈ [0.0, 1.0, 0.0]
    @test za[3, 1, :] ≈ [0.75, 0.25, 0.0]
    @test za[1, 2, :] ≈ [0.75, 0.25, 0.0]
    @test za[3, 4, :] ≈ [0.75, 0.25, 0.0]
    @test all(isfinite, za) && all(isfinite, yb)
    za, _ = ot.ot_joint(missing_in_B, 0.5, 1.0, 1.0)
    @test 0.0 < sum(za[3, 1, :]) < 0.99

    missing_in_A = Instance(
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        reshape([1, 1, 1, 2, 1, 1, 2, 2, 3, 3], :, 1),
        [1, 1, 1, 2, 1, 1, 1, 1, 1, 1], 1:4, fill(1, 10), 1:3, Euclidean()
    )
    za, yb = ot.ot_joint(missing_in_A, 0.0, 0.0, 1.0)
    @test za[3, 1, :] ≈ [1.0, 0.0, 0.0]
    @test yb[1, :, 1] ≈ [1.0, 0.0, 0.0, 0.0]
    @test yb[2, :, 1] ≈ [0.0, 1.0, 0.0, 0.0]
    @test yb[3, :, 1] ≈ [0.75, 0.25, 0.0, 0.0]
    @test all(isfinite, za) && all(isfinite, yb)
    _, yb = ot.ot_joint(missing_in_A, 0.5, 1.0, 1.0)
    @test 0.0 < sum(yb[3, :, 1]) < 0.99

    disjoint = Instance(base, reshape([1, 1, 2, 2], :, 1),
        [1, 1, 1, 1], 1:4, [1, 1, 1, 1], 1:3, Euclidean())
    za, yb = ot.ot_joint(disjoint, 0.0, 0.0, 1.0)
    @test za[1, 1, :] ≈ [1.0, 0.0, 0.0]
    @test yb[2, :, 1] ≈ [1.0, 0.0, 0.0, 0.0]

    continuous = DataFrame(
        database = [1, 1, 1, 1, 2, 2, 2, 2],
        X1 = [1.1, 1.1, 2.1, 2.1, 1.1, 1.1, 2.1, 2.1],
        Y = [1, 2, 1, 2, 1, 2, 1, 2],
        Z = [1, 2, 1, 2, 1, 2, 1, 2],
    )
    result = otrecod(continuous, JointOTWithinBase(alpha = 0.0, lambda = 0.0))
    @test length(result.yb_pred) == length(result.za_pred) == 4
    continuous.X1 = [1.1, 1.1, 2.1, 2.1, 1.1, 1.1, 1.1, 1.1]
    result = otrecod(continuous, JointOTWithinBase(alpha = 0.0, lambda = 0.0))
    @test length(result.yb_pred) == length(result.za_pred) == 4
    continuous.X1 = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 2.1, 2.1]
    result = otrecod(continuous, JointOTWithinBase(alpha = 0.0, lambda = 0.0))
    @test length(result.yb_pred) == length(result.za_pred) == 4
    continuous.X1 = [1.1, 1.1, 1.1, 1.1, 2.1, 2.1, 2.1, 2.1]
    result = otrecod(continuous, JointOTWithinBase(alpha = 0.1, lambda = 0.1))
    @test length(result.yb_pred) == length(result.za_pred) == 4
end

@testset "Cross-entropy weights" begin
    data = DataFrame(
        database = [1, 1, 1, 1, 2, 2, 2, 2],
        X1 = [1, 1, 2, 2, 1, 1, 2, 2],
        Y = [1, 2, 1, 2, 1, 2, 1, 2],
        Z = [1, 2, 1, 2, 1, 2, 1, 2],
    )
    for method in (JointOTBetweenBases, JointOTBetweenBasesDiscreteOrdered)
        @test_throws ArgumentError otrecod(data, method(alpha1 = -1.0))
        @test_throws ArgumentError otrecod(data, method(alpha2 = Inf))
        @test_throws ArgumentError otrecod(data, method(alpha1 = NaN))
        default = otrecod(data, method(iterations = 2))
        weight = 1 / Float32(-log(1.0e-12))
        explicit = otrecod(data, method(iterations = 2, alpha1 = weight, alpha2 = weight))
        @test default.yb_pred == explicit.yb_pred
        @test default.za_pred == explicit.za_pred
    end
    data.X1 = Float64.(data.X1) .+ 0.5
    for method in (JointOTBetweenBasesWithPredictors, JointOTBetweenBasesWithoutOutcomes)
        @test_throws ArgumentError otrecod(data, method(alpha1 = Inf))
        @test_throws ArgumentError otrecod(data, method(alpha2 = -1.0))
        @test_throws ArgumentError otrecod(data, method(alpha2 = NaN))
    end
end

include(joinpath(@__DIR__, "..", "experiments", "tune_alpha_weights.jl"))

@testset "Alpha weight validation search" begin
    params = DataParameters(nA = 40, nB = 40)
    options = (
        params = params, nsimulations = 1, validation_seed = 314159,
        alpha_grid = [0.001, 10.0], iterations = 2, epochs = 1, batchsize = 32,
    )
    scores = tune_alpha_weights(; options...)
    @test size(scores, 1) == 8
    @test scores == tune_alpha_weights(; options...)
    @test all(x -> 0 <= x <= 1, scores.accuracy)
    @test_throws ArgumentError tune_alpha_weights(; nsimulations = 0)
    @test_throws ArgumentError tune_alpha_weights(; iterations = 1)
    @test_throws ArgumentError tune_alpha_weights(; epochs = 0)

    scenario1 = scores[scores.scenario .== 1, :]
    score(a, b) = only(scenario1.accuracy[(scenario1.alpha1 .== a) .& (scenario1.alpha2 .== b)])
    @test score(0.001, 0.001) != score(0.001, 10.0)
    @test score(0.001, 0.001) != score(10.0, 0.001)

    for scenario in 1:2
        Random.seed!(314159 + scenario - 1)
        generator = DiscreteDataGenerator(params; scenario = scenario)
        data = generate(generator)
        method = JointOTBetweenBasesWithPredictors(
            alpha1 = 0.001, alpha2 = 10.0, reg = 0.0, iterations = 2,
            epochs = 1, batchsize = 32
        )
        training_seeds = rand(MersenneTwister(314159 + scenario - 1), UInt64, 1)
        Random.seed!(training_seeds[1])
        expected = accuracy(otrecod(data, method))
        row = only(eachrow(scores[
            (scores.scenario .== scenario) .& (scores.alpha1 .== 0.001) .&
                (scores.alpha2 .== 10.0), :
        ]))
        @test [row.accuracy_yb, row.accuracy_za, row.accuracy] ≈ collect(expected)
    end

    io = IOBuffer()
    print_validation_scores(scores, 314159; io = io)
    output = String(take!(io))
    @test occursin("Scenario 1", output) && occursin("Scenario 2", output)
    @test length(collect(eachmatch(r"Selected:", output))) == 2
    @test occursin("independent simulations", output)
    for scenario in 1:2
        rows = scores[scores.scenario .== scenario, :]
        best = rows[argmax(rows.accuracy), :]
        @test occursin(
            @sprintf("Selected: alpha1=%.3g alpha2=%.3g (validation accuracy=%.6f)",
                best.alpha1, best.alpha2, best.accuracy),
            output,
        )
    end
end

params = DataParameters(nA = 1000, nB = 1000)

rng = DiscreteDataGenerator(params, n = 1000, scenario = 1)

data = generate(rng)


@testset "discrete covariables JointOTBetweenBasesDiscreteOrdered" begin

    @test all(accuracy(otrecod(data, JointOTBetweenBasesDiscreteOrdered())) .> 0.0)

end

@testset "check data generator" begin

    @test sort(unique(data.Y)) ≈ [1, 2, 3, 4]
    @test sort(unique(data.Z)) ≈ [1, 2, 3]

end

@testset "discrete covariables JointOTWithinBase" begin

    result = otrecod(data, JointOTWithinBase())
    @test all(accuracy(result) .> 0.5)

end

@testset "discrete covariables SimpleLearning" begin

    result = otrecod(data, SimpleLearning())
    @test all(accuracy(result) .> 0.0)

end

@testset "SimpleLearning one-hot categories" begin

    x = [1 2; 5 4]
    encoded = OptimalTransportDataIntegration.onehot(x)
    @test size(encoded) == (9, 2)
    @test encoded[5, 2] == 1

    encoded_subset = OptimalTransportDataIntegration.onehot(x[1:1, :]; nlevels = [5, 4])
    @test size(encoded_subset, 1) == 9

end

@testset "SimpleLearning nonlinear hidden layer" begin

    model = OptimalTransportDataIntegration._simple_learning_model(1, 1, 1)
    hidden, output = model.layers
    hidden.weight .= 1
    hidden.bias .= 0
    output.weight .= 1
    output.bias .= 0

    @test model(reshape(Float32[-1, 1], 1, :)) == reshape(Float32[0, 1], 1, :)

end

data = CSV.read(joinpath(@__DIR__, "data_good.csv"), DataFrame)

@testset "Unbalanced method with good data" begin

    @time result = otrecod(data, JointOTBetweenBases())
    @test all(accuracy(result) .> 0.5)

end

@testset "Balanced method with good data" begin

    @time result = otrecod(data, JointOTBetweenBases(reg_m1 = 0.0, reg_m2 = 0.0))
    println(accuracy(result))

end

data = CSV.read(joinpath(@__DIR__, "data_bad.csv"), DataFrame)

@testset "Unbalanced method with bad data" begin

    @time result = otrecod(data, JointOTBetweenBases())
    println(accuracy(result))

end

@testset "Balanced method with bad data" begin

    @time result = otrecod(data, JointOTBetweenBases(reg_m1 = 0.0, reg_m2 = 0.0))
    println(accuracy(result))

end

rng = ContinuousDataGenerator(params, scenario = 1)
data = generate(rng)

@testset "continuous covariables SimpleLearning" begin

    result = otrecod(data, SimpleLearning(epochs = 1, batchsize = 128))
    @test length(result.yb_pred) == count(==(2), data.database)
    @test length(result.za_pred) == count(==(1), data.database)
    @test all(x -> x in 1:4, result.yb_pred)
    @test all(x -> x in 1:3, result.za_pred)

end

@testset "JointOTWithinBase" begin

    @test all(accuracy(otrecod(data, JointOTWithinBase())) .> 0.0)

end

@testset "JointOTBetweenBasesWithPredictors" begin

    @test all(accuracy(otrecod(data, JointOTBetweenBasesWithPredictors())) .> 0.0)

end

@testset "JointOTBetweenBasesWithoutOutcomes" begin

    @test all(accuracy(otrecod(data, JointOTBetweenBasesWithoutOutcomes(reg = 0.0))) .> 0.0)

end

@testset "JointOTBetweenBasesJDOT" begin

    @test all(accuracy(otrecod(data, JointOTBetweenBasesJDOT(reg = 0.0))) .> 0.0)

end

@testset "JointOTDABetweenBasesCovariables" begin

    @test all(accuracy(otrecod(data, JointOTDABetweenBasesCovariables(reg = 0.0))) .> 0.0)

end

@testset "JointOTDABetweenBasesOutcomes" begin

    @test all(accuracy(otrecod(data, JointOTDABetweenBasesOutcomes(reg = 0.0))) .> 0.0)

end

@testset "JointOTDABetweenBasesOutcomesWithPredictors" begin

    @test all(accuracy(otrecod(data, JointOTDABetweenBasesOutcomesWithPredictors(reg = 0.0))) .> 0.0)

end


@testset "Aqua.jl" begin
    Aqua.test_all(OptimalTransportDataIntegration)
end

@testset "doctests" begin
    DocMeta.setdocmeta!(
        OptimalTransportDataIntegration,
        :DocTestSetup,
        :(using OptimalTransportDataIntegration);
        recursive = true,
    )
    doctest(
        OptimalTransportDataIntegration;
        doctestfilters = [
            r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
            r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
            r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
        ],
    )
end

using Aqua
using CSV
using DataFrames
using Documenter
using OptimalTransportDataIntegration
using Test

params = DataParameters(nA = 1000, nB = 1000)
    
rng = DataGenerator(params, n = 1000)

data = generate_data(rng)

@testset "check data generator" begin

    @test sort(unique(data.Y)) ≈ [1, 2, 3, 4]
    @test sort(unique(data.Z)) ≈ [1, 2, 3]

end

@testset "JointOTWithinBase method" begin

    result = otrecod(data, JointOTWithinBase()) 
    @test all(accuracy(result) .> 0.8)

end

@testset "SimpleLearning method" begin

    result = otrecod(data, SimpleLearning())
    @test all(accuracy(result) .> 0.8)

end

data = CSV.read(joinpath(@__DIR__, "data_good.csv"), DataFrame)

@testset "Unbalanced method with good data" begin

    @time result = otrecod(data, JointOTBetweenBases())
    println(accuracy(result))

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

@testset "Aqua.jl" begin
    Aqua.test_deps_compat(OptimalTransportDataIntegration)
end

@testset "doctests" begin
    DocMeta.setdocmeta!(OptimalTransportDataIntegration, :DocTestSetup, :(using OptimalTransportDataIntegration); recursive=true)
    doctest(
        OptimalTransportDataIntegration;
        doctestfilters=[
            r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
            r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
            r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
        ],
    )
end

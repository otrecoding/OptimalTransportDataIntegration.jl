using Aqua
using CSV
using DataFrames
using Documenter
using OptimalTransportDataIntegration
using Test


@testset "otrecod generated data" begin

    params = DataParameters(nA = 1000, nB = 1000)
    
    data = generate_xcat_ycat(params)
    
    @test sort(unique(data.Y)) ≈ [1, 2, 3, 4]
    @test sort(unique(data.Z)) ≈ [1, 2, 3]
    
    @test otrecod(data, OTjoint()) > 0.8
    @test otrecod(data, SimpleLearning()) > 0.5

end
    
@testset "otrecod data with all levels in Y and Z" begin

    data = CSV.read(joinpath(@__DIR__, "data_good.csv"), DataFrame)
    @time est = otrecod(data, UnbalancedModality())
    println(est)
    @time est = otrecod(data, UnbalancedModality(reg_m1 = 0.0, reg_m2 = 0.0))
    println(est)

end

@testset "otrecod data with missing levels in Y or Z" begin

    data = CSV.read(joinpath(@__DIR__, "data_bad.csv"), DataFrame)
    @time est = otrecod(data, UnbalancedModality())
    println(est)
    
    @time est = otrecod(data, UnbalancedModality(reg_m1 = 0.0, reg_m2 = 0.0))
    println(est)

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

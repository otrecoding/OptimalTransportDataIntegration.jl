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
    
    yb, za = otrecod(data, OTjoint()) 
    @test all(accuracy(data, yb, za) .> 0.8)
    yb, za = otrecod(data, SimpleLearning())
    @test all(accuracy(data, yb, za) .> 0.8)

end
    
@testset "otrecod data with all levels in Y and Z" begin

    data = CSV.read(joinpath(@__DIR__, "data_good.csv"), DataFrame)
    @time yb, za = otrecod(data, UnbalancedModality())
    println(accuracy(data, yb, za))
    @time yb, za = otrecod(data, UnbalancedModality(reg_m1 = 0.0, reg_m2 = 0.0))
    println(accuracy(data, yb, za))

end

@testset "otrecod data with missing levels in Y or Z" begin

    data = CSV.read(joinpath(@__DIR__, "data_bad.csv"), DataFrame)
    @time yb, za = otrecod(data, UnbalancedModality())
    println(accuracy(data, yb, za))
    
    @time yb, za = otrecod(data, UnbalancedModality(reg_m1 = 0.0, reg_m2 = 0.0))
    println(accuracy(data, yb, za))

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

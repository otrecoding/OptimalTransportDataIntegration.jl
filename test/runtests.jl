using Aqua
using CSV
using DataFrames
using OptimalTransportDataIntegration
using Test

@testset "Aqua.jl" begin
    Aqua.test_deps_compat(OptimalTransportDataIntegration)
end

params = DataParameters(nA = 1000, nB = 500)

data = generate_xcat_ycat(params)

@test sort(unique(data.Y)) ≈ [1, 2, 3, 4]
@test sort(unique(data.Z)) ≈ [1, 2, 3]

@test otrecod(data, OTjoint()) > 0.8
@test otrecod(data, SimpleLearning()) > 0.5

data = CSV.read(joinpath(@__DIR__, "data_good.csv"), DataFrame)
@time est = otrecod(data, UnbalancedModality())
println(est)
@time est = otrecod(data, UnbalancedModality(reg_m = 0.0))
println(est)

data = CSV.read(joinpath(@__DIR__, "data_bad.csv"), DataFrame)
@time est = otrecod(data, UnbalancedModality())
println(est)
@time est = otrecod(data, UnbalancedModality(reg_m = 0.0))
println(est)

using DataFrames
using OptimalTransportDataIntegration
using Printf
using Random
using Statistics

function tune_alpha_weights(;
        params = DataParameters(),
        nsimulations = 100,
        validation_seed = 20261123,
        alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0],
        iterations = 10,
        epochs = 1000,
        batchsize = 64,
        hidden_layer_size = 10,
        learning_rate = 0.01,
    )
    nsimulations > 0 || throw(ArgumentError("nsimulations must be positive"))
    iterations >= 2 || throw(ArgumentError("iterations must be at least 2 for weights to affect predictions"))
    epochs > 0 || throw(ArgumentError("epochs must be positive"))
    !isempty(alpha_grid) && all(x -> isfinite(x) && x >= 0, alpha_grid) ||
        throw(ArgumentError("alpha_grid must contain finite nonnegative weights"))

    scores = DataFrame(
        scenario = Int[], alpha1 = Float64[], alpha2 = Float64[],
        accuracy_yb = Float64[], accuracy_za = Float64[], accuracy = Float64[]
    )

    for scenario in 1:2
        Random.seed!(validation_seed + scenario - 1)
        generator = DiscreteDataGenerator(params; scenario = scenario)
        validation_data = [generate(generator) for _ in 1:nsimulations]
        training_seeds = rand(MersenneTwister(validation_seed + scenario - 1), UInt64, nsimulations)

        for alpha1 in alpha_grid, alpha2 in alpha_grid
            method = JointOTBetweenBasesWithPredictors(;
                alpha1 = alpha1, alpha2 = alpha2, reg = 0.0, iterations = iterations,
                epochs = epochs, batchsize = batchsize, hidden_layer_size = hidden_layer_size,
                learning_rate = learning_rate
            )
            metrics = Tuple{Float64, Float64, Float64}[]
            for (i, data) in enumerate(validation_data)
                Random.seed!(training_seeds[i])
                push!(metrics, accuracy(otrecod(data, method)))
            end
            push!(scores, (
                scenario, alpha1, alpha2,
                mean(first.(metrics)), mean(getindex.(metrics, 2)), mean(last.(metrics))
            ))
        end
    end

    return scores
end

function print_validation_scores(scores, validation_seed; io = stdout)
    println(io, "Validation accuracy only; use independent simulations for final evaluation.")
    println(io, "Validation seeds: scenario 1 = $validation_seed, scenario 2 = $(validation_seed + 1)")
    for scenario in 1:2
        rows = scores[scores.scenario .== scenario, :]
        println(io, "\nScenario $scenario")
        println(io, "alpha1\talpha2\taccuracy_yb\taccuracy_za\toverall_accuracy")
        for row in eachrow(rows)
            @printf(io, "%.3g\t%.3g\t%.6f\t%.6f\t%.6f\n",
                row.alpha1, row.alpha2, row.accuracy_yb, row.accuracy_za, row.accuracy)
        end
        best = rows[argmax(rows.accuracy), :]
        @printf(io, "Selected: alpha1=%.3g alpha2=%.3g (validation accuracy=%.6f)\n",
            best.alpha1, best.alpha2, best.accuracy)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) <= 3 || error("Usage: julia --project experiments/tune_alpha_weights.jl [nsimulations] [validation_seed] [sample_size]")
    nsimulations = 100
    validation_seed = 20261123
    sample_size = 1000
    if length(ARGS) >= 1
        nsimulations = parse(Int, ARGS[1])
    end
    if length(ARGS) >= 2
        validation_seed = parse(Int, ARGS[2])
    end
    if length(ARGS) >= 3
        sample_size = parse(Int, ARGS[3])
    end
    sample_size >= 2 || throw(ArgumentError("sample_size must be at least 2"))
    params = DataParameters(nA = sample_size, nB = sample_size)
    scores = tune_alpha_weights(; params = params, nsimulations = nsimulations, validation_seed = validation_seed)
    print_validation_scores(scores, validation_seed)
end

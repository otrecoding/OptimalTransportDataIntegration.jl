# Parameter optimization

The repository includes **simulation-based grid searches** for a small subset of
method parameters. They are experiment scripts, not a general-purpose tuner:
they generate data with known outcomes, run each candidate configuration, and
write the resulting accuracies to CSV files.

## Parameters covered by the experiments

| Method and data | Parameters varied | Candidate values |
|---|---|---|
| `JointOTWithinBase`, discrete or continuous covariates | `alpha`, `lambda` | `alpha = 0:0.1:2`, `lambda = 0:0.1:1` |
| `JointOTBetweenBases`, discrete covariates | `reg`, with `reg_m1 = reg_m2` | `reg = [0.001, 0.01, 0.1, 1.0]`; each marginal relaxation in `[0.01, 0.05, 0.1, 0.25, 0.5, 0.75]` |
| `JointOTBetweenBasesWithPredictors`, continuous covariates | `reg`, with `reg_m1 = reg_m2` | Same values as the discrete between-bases search |

These are the grids in [`experiments/best_parameters.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/best_parameters.jl).
That script evaluates both generator scenarios and, by default, 100 generated
datasets per scenario and data type. It writes
`best_parameters_discrete.csv` and `best_parameters_continuous.csv` in the
current working directory. The CSV rows contain per-dataset scores; the script
does not itself aggregate them or select a winner.

[`experiments/best_reg_m.jl`](https://github.com/otrecoding/OptimalTransportDataIntegration.jl/blob/main/experiments/best_reg_m.jl)
provides a narrower sensitivity experiment for continuous
`JointOTBetweenBasesWithPredictors`: it fixes `reg = 0.001`, varies equal
marginal relaxations `reg_m1 = reg_m2` from `0.0001` to `0.05`, and evaluates
four covariate-mean shifts with 100 simulations per shift. It writes
`best_reg_m_values_with_different_mb.csv`.

Both scripts run their full experiment when executed, with simulation counts
set in the source; they do not expose command-line options for smaller runs.
For example, the following snippet compares a few within-base settings on one
generated dataset:

```julia
using OptimalTransportDataIntegration

params = DataParameters(nA = 200, nB = 200)
data = generate(ContinuousDataGenerator(params; scenario = 1))

for alpha in (0.0, 1.0), lambda in (0.1, 1.0)
    scores = accuracy(otrecod(data, JointOTWithinBase(alpha = alpha, lambda = lambda)))
    @show alpha lambda scores
end
```

Run the complete experiment scripts from the repository root with:

```sh
julia --project experiments/best_parameters.jl
julia --project experiments/best_reg_m.jl
```

## Reading the results

`accuracy(result)` returns accuracies for the imputed `Y` values in database B,
the imputed `Z` values in database A, and the combined predictions, in that
order. In these experiments, the generators retain the outcomes needed to
compute ground-truth accuracy. The first script records these three values for
each generated dataset, alongside the method, scenario, and parameter values.
Compare candidates across the same simulated scenarios and inspect the two
outcome-specific accuracies as well as the combined score.

The exported `best_parameters(method, variable, scenario)` function returns
scenario-specific presets used in the simulation documentation; despite its
name, it does not run a search. For example,
`best_parameters(:within, :continuous, 1)` returns `(1.0, 1.0)` for
`(alpha, lambda)`, while `best_parameters(:between, :continuous, 2)` returns
`(0.001, 0.5)` for `(reg, reg_m)`. These values are presets, not a general
recommendation for arbitrary datasets.

## Scope and limitations

- The scripts cover only the parameter combinations listed above; they do not
  tune every field of these methods or parameters for every method.
- The between-bases sweeps tie `reg_m1` and `reg_m2` to the same value.
- The scripts score predictions on the same generated datasets used for each
  run. They therefore document simulation accuracy, not an independent
  validation or model-selection protocol for real data.
- The cross-entropy weights `alpha1` and `alpha2` are not searched by these
  scripts. Their tuning criterion, search ranges, and an independent
  validation protocol remain to be clarified with Valérie; no tuning guidance
  for those weights is prescribed here.

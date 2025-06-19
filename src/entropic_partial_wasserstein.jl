using LinearAlgebra

export entropic_partial_wasserstein

@doc raw"""
    entropic_partial_wasserstein(a, b, M, reg, m=nothing, numItermax=1000, 
        stopThr=1e-100, verbose=false, log=false)

This function is a Julia translation of the function
[`entropic_partial_wasserstein`](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.entropic_partial_wasserstein) in the
Python Optimal Transport package. 

Solves the partial optimal transport problem
and returns the OT plan

The function considers the following problem:

```math
    \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma,
             \mathbf{M} \rangle_F + \mathrm{reg} \cdot\Omega(\gamma)
```

```math
\begin{aligned}
     s. t.  \qquad \gamma \mathbf{1} &  \leq \mathbf{a} \\
     \gamma^T \mathbf{1} &\leq \mathbf{b} \\
     \gamma &\geq 0 \\
         \mathbf{1}^T \gamma^T \mathbf{1} = m
         &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\} \\
\end{aligned}
```

where :

- ``\mathbf{M}`` is the metric cost matrix
- ``\Omega``  is the entropic regularization term,
  ``\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})``
- ``\mathbf{a}`` and :math:`\mathbf{b}` are the sample weights
- ``m`` is the amount of mass to be transported

The formulation of the problem has been proposed in
:ref:`[3] <references-entropic-partial-wasserstein>` (prop. 5)


# Parameters

- `a` : Array (dim_a,) Unnormalized histogram of dimension `dim_a`
- `b` : Array (dim_b,) Unnormalized histograms of dimension `dim_b`
- `M` : Array (dim_a, dim_b) cost matrix
- `reg` : Float Regularization term > 0
- `m` : Float, optional Amount of mass to be transported
- `numItermax` : Int, optional Max number of iterations
- `stopThr` : Float, optional Stop threshold on error (>0)
- `verbose` : Bool, optional Print information along iterations
- `log` : Bool, optional record log if True


# Returns

- `gamma` : (dim_a, dim_b) ndarray
    Optimal transportation matrix for the given parameters
- `log` : Dict
    log dictionary returned only if `log` is `True`


# Examples

```jldoctest
julia> a = [.1, .2];

julia> b = [.1, .1];

julia> M = [0. 1.; 2. 3.];

julia> round.(entropic_partial_wasserstein(a, b, M, 1, m = 0.1), digits=2)
2×2 Matrix{Float64}:
 0.06  0.02
 0.01  0.0
```

# References

- [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
   (2015). Iterative Bregman projections for regularized transportation
   problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.


"""
function entropic_partial_wasserstein(
        a,
        b,
        M,
        reg;
        m = nothing,
        numItermax = 1000,
        stopThr = 1.0e-100,
        verbose = false,
        log = false,
    )

    dim_a, dim_b = size(M)
    dx = ones(eltype(a), dim_a)
    dy = ones(eltype(b), dim_b)

    if length(a) == 0
        a = ones(eltype(a), dim_a) ./ dim_a
    end
    if length(b) == 0
        b = ones(eltype(b), dim_b) ./ dim_b
    end

    if isnothing(m)
        m = min(sum(a), sum(b))
    end
    if m < 0
        @error "Problem infeasible. Parameter m should be greater" " than 0."
    end
    if m > min(sum(a), sum(b))
        @error "Problem infeasible. Parameter m should lower or equal than min(|a|_1, |b|_1)."
    end

    log && (log_e = Dict{Any, Any}("err" => []))

    K = exp.(-M ./ reg)
    K .= K .* m ./ sum(K)

    err, cpt = 1, 0
    q1 = ones(eltype(K), size(K))
    q2 = ones(eltype(K), size(K))
    q3 = ones(eltype(K), size(K))
    Kprev = similar(K)
    K1 = similar(K)
    K1prev = similar(K1)
    K2 = similar(K1)
    K2prev = similar(K2)

    while err > stopThr && cpt < numItermax
        Kprev .= K
        K .= K .* q1
        K1 .= Diagonal(min.(a ./ vec(sum(K, dims = 2)), dx)) * K
        q1 .= q1 .* Kprev ./ K1
        K1prev .= K1
        K1 .= K1 .* q2
        K2 .= K1 * Diagonal(min.(b ./ vec(sum(K1, dims = 1)), dy))
        q2 .= q2 .* K1prev ./ K2
        K2prev .= K2
        K2 .= K2 .* q3
        K .= K2 .* (m ./ sum(K2))
        q3 .= q3 .* K2prev ./ K

        if any(isnan.(K)) || any(isinf.(K))
            @warn "Warning: numerical errors at iteration $cpt"
            break
        end
        if cpt % 10 == 0
            err = norm(Kprev .- K)
            log && push!(log_e["err"], err)
            if verbose
                if cpt % 200 == 0
                    println("It. Err - $cpt $err")
                end
            end
        end
        cpt = cpt + 1
    end
    if log
        log_e["partial_w_dist"] = sum(M .* K)
        return K, log_e
    else
        return K
    end
end

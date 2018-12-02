"""
Solve a least square problem for a set of FixedEffects

### Arguments
* `y` : an `AbstractVector{Float64}` or an `AbstractMatrix{Float64}`
* `fes`: A Vector of `FixedEffect`
* `weights`: Weights
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :lsmr_threads, :lsmr_parallel, :qr and :cholesky (factorization methods)
* `maxiter` : Maximum number of iterations
* `tol` : tolerance


### Returns
* `res` : the residual of the least square problem
* `iterations`: number of iterations
* `converged`: did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10, 5)
solve_residuals!(x, [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_residuals!(y::AbstractVector{Float64}, fes::Vector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{Float64}(length(y))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    for fe in fes
        if ismissing(fe)
            error("Some FixedEffect has a missing value for reference or interaction")
        end
    end
    sqrtw = sqrt.(weights.values)
    y .= y .* sqrtw
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    y, iteration, converged = solve_residuals!(y, fep; maxiter = maxiter, tol = tol)
    y .= y ./ sqrtw
    return y, iteration, converged
end

function solve_residuals!(y::AbstractMatrix{Float64}, fes::Vector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{Float64}(size(y, 1)));  method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    for fe in fes
        if ismissing(fe)
            error("Some FixedEffect has a missing value for reference or interaction")
        end
    end
    sqrtw = sqrt.(weights.values)
    y .= y .* sqrtw
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    y, iterations, convergeds = solve_residuals!(y, fep; maxiter = maxiter, tol = tol)
    y .= y ./ sqrtw
    return y, iterations, convergeds
end


"""
Solve a least square problem for a set of FixedEffects

### Arguments
* `y` : an `AbstractVector{Float64}` 
* `fes`: A Vector of `FixedEffect`
* `weights`: Weights
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)
* `maxiter` : Maximum number of iterations
* `tol` : tolerance


### Returns
* `b` : the solution of the least square problem
* `iterations`: number of iterations
* `converged`: did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
X = rand(10, 5)
solve_coefficients!(x, [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_coefficients!(y, fes::Vector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{Float64}(length(y))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    for fe in fes
        if ismissing(fe)
            error("Some FixedEffect has a missing value for reference or interaction")
        end
    end

    sqrtw = sqrt.(weights.values)
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    y .= y .* sqrtw
    newfes, iteration, converged = solve_coefficients!(y, fep; maxiter = maxiter, tol = tol)
    return newfes, iteration, converged
end




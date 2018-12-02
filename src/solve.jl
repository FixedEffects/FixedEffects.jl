"""

Solve a least square problem for a set of FixedEffects

`solve_residuals!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector{Float64}` or a `AbstractMatrix{Float64}`
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :qr and :cholesky
* `maxiter` : Maximum number of iterations
* `tol` : Tolerance


### Returns
* `res` :  Residual of the least square problem
* `iterations`: Number of iterations
* `converged`: Did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
solve_residuals!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
solve_residuals!(rand(10, 5), [FixedEffect(p1), FixedEffect(p2)])
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

function solve_residuals!(y::AbstractMatrix{Float64}, fes::Vector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{Float64}(size(y, 1)));  method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
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

`solve_coefficients!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector{Float64}` 
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :qr and :cholesky
* `maxiter` : Maximum number of iterations
* `tol` : Tolerance


### Returns
* `b` : Solution of the least square problem
* `iterations`: Number of iterations
* `converged`: Did the algorithm converge?

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
solve_coefficients!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
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




"""

Solve a least square problem for a set of FixedEffects

`solve_residuals!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector` or a `AbstractMatrix`
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
function solve_residuals!(y::Union{AbstractVector, AbstractMatrix}, fes::Vector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{eltype(y)}(size(y, 1))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    sqrtw = sqrt.(weights.values)
    y .= y .* sqrtw
    fep = FixedEffectMatrix(fes, sqrtw, Val{method})
    y, iteration, converged = solve_residuals!(y, fep; maxiter = maxiter, tol = tol)
    y .= y ./ sqrtw
    return y, iteration, converged
end

"""
Solve a least square problem for a set of FixedEffects

`solve_coefficients!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector` 
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
function solve_coefficients!(y::AbstractVector, fes::Vector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{eltype(y)}(length(y))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    sqrtw = sqrt.(weights.values)
    y .= y .* sqrtw
    fep = FixedEffectMatrix(fes, sqrtw, Val{method})
    newfes, iteration, converged = solve_coefficients!(y, fep; maxiter = maxiter, tol = tol)
    return newfes, iteration, converged
end




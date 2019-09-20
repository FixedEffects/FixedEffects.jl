
# this type must defined solve_residuals!, solve_coefficients!
abstract type AbstractFixedEffectMatrix end

"""
Solve a least square problem for a set of FixedEffects

`solve_residuals!(y, fes, weights; method = :lsmr, maxiter = 10000, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector` or an `AbstractMatrix`
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :lsmr_gpu (available only if `CuArrays` is loaded), :qr and :cholesky.
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
```
"""
function solve_residuals!(y::Union{AbstractVector, AbstractMatrix}, fes::AbstractVector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{eltype(y)}(size(y, 1))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    feM = FixedEffectMatrix(fes, sqrt.(weights.values), Val{method})
    y, iteration, converged = solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

function solve_residuals!(X::AbstractMatrix, feM::AbstractFixedEffectMatrix; kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    for x in eachcol(X)
        _, iteration, converged = solve_residuals!(x, feM; kwargs...)
        push!(iterations, iteration)
        push!(convergeds, converged)
    end
    return X, iterations, convergeds
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
Fixed effects are generally not unique. We standardize the solution 
in the following way: the mean of fixed effects within connected components is zero
(except for the first).
This gives the unique solution in the case of two fixed effects.

### Examples
```julia
using  FixedEffects
p1 = repeat(1:5, inner = 2)
p2 = repeat(1:5, outer = 2)
x = rand(10)
solve_coefficients!(rand(10), [FixedEffect(p1), FixedEffect(p2)])
```
"""
function solve_coefficients!(y::AbstractVector, fes::AbstractVector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{eltype(y)}(length(y))); method::Symbol = :lsmr, maxiter::Integer = 10000, tol::Real = 1e-8)
    any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
    feM = FixedEffectMatrix(fes, sqrt.(weights.values), Val{method})
    solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end
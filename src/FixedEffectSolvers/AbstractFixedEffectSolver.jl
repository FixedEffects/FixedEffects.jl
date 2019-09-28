##############################################################################
## 
## AbstractFixedEffectSolver
##
## this type must defined solve_residuals!, solve_coefficients!
##
##############################################################################
abstract type AbstractFixedEffectSolver{T} end


"""
Solve a least square problem for a set of FixedEffects

`solve_residuals!(y, fes, weights; method = :lsmr, maxiter = 10000, double_precision = true, tol = 1e-8)`

### Arguments
* `y` : A `AbstractVector` or an `AbstractMatrix`
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :lsmr_gpu (requires `CuArrays`. Use the option `double_precision = false` to use `Float32` on the GPU).
* `maxiter` : Maximum number of iterations
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol` : Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.

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
function solve_residuals!(y::Union{AbstractVector{<: Number}, AbstractMatrix{<: Number}}, fes::AbstractVector{<: FixedEffect}, weights::AbstractWeights = Weights(Ones{eltype(y)}(size(y, 1))); 
	method::Symbol = :lsmr, maxiter::Integer = 10000, 
	double_precision::Bool = eltype(y) == Float64, tol::Real = double_precision ? 1e-8 : 1e-6)
	any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, sqrt.(weights.values), Val{method})
	solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

"""
Solve a least square problem for a set of FixedEffects

`solve_coefficients!(y, fes, weights; method = :lsmr, maxiter = 10000, double_precision = true, tol = 1e-8)`
d
### Arguments
* `y` : A `AbstractVector` 
* `fes`: A `Vector{<:FixedEffect}`
* `weights`: A `AbstractWeights`
* `method` : A `Symbol` for the method. Choices are :lsmr, :lsmr_threads, :lsmr_parallel, :lsmr_gpu (requires `CuArrays`. Use the option `double_precision = false` to use `Float32` on the GPU).
* `maxiter` : Maximum number of iterations
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol` : Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.

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
function solve_coefficients!(y::AbstractVector{<: Number}, fes::AbstractVector{<: FixedEffect}, weights::AbstractWeights  = Weights(Ones{eltype(y)}(length(y))); 
	method::Symbol = :lsmr, maxiter::Integer = 10000,
	double_precision::Bool = eltype(y) == Float64, tol::Real = double_precision ? 1e-8 : 1e-6)
	any(ismissing.(fes)) && error("Some FixedEffect has a missing value for reference or interaction")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, sqrt.(weights.values), Val{method})
	solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end

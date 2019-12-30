##############################################################################
## 
## AbstractFixedEffectSolver
##
## this type must defined solve_residuals!, solve_coefficients!
##
##############################################################################
abstract type AbstractFixedEffectSolver{T} end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Union{Type{Val{:lsmr}}, Type{Val{:lsmr_threads}}, Type{Val{:lsmr_cores}}}) where {T}
	@warn ":lsmr, :lsmr_threads, and :lsmr_cores are deprecated (FixedEffects is now always multi-threaded)"
	AbstractFixedEffectSolver{T}(fes, weights, Val{:cpu})
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:lsmr_gpu}}) where {T}
	@warn ":lsmr_gpu is deprecated,  use :gpu"
	AbstractFixedEffectSolver{T}(fes, weights, Val{:gpu})
end

"""
`solve_residuals!(y, fes, w; method = :cpu, double_precision = true, tol = 1e-8, maxiter = 10000, )`

Returns ``y_i - X_i'\\beta`` where ``\\beta = argmin_{b} \\sum_i y_i - X_i'b``, where `X` denotes the matrix of fixed effects `fes`.

### Arguments
* `y` : A `AbstractVector` or A `AbstractMatrix`
* `fes`: A `Vector{<:FixedEffect}`
* `w`: A vector of weights, i.e. `AbstractWeights`
* `method` : A `Symbol` for the method. Default is :cpu. The option :gpu requires `CuArrays` (in this case, it is recommanded to use the option `double_precision = false`).
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol` : Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `maxiter` : Maximum number of iterations

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
function solve_residuals!(y::Union{AbstractVector{<: Number}, AbstractMatrix{<: Number}}, fes::AbstractVector{<: FixedEffect}, w::AbstractWeights = Weights(Ones{eltype(y)}(size(y, 1))); 
	method::Symbol = :cpu, double_precision::Bool = eltype(y) == Float64, 
	tol::Real = double_precision ? 1e-8 : 1e-6, maxiter::Integer = 10000)
	any((length(fe) != size(y, 1) for fe in fes)) && throw("FixedEffects must have the same length as y")
	any(ismissing.(fes)) && throw("FixedEffects must not have missing values")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, w, Val{method})
	solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

"""
Solve a least square problem for a set of FixedEffects

`solve_coefficients!(y, fes, w; method = :cpu, maxiter = 10000, double_precision = true, tol = 1e-8)`

Returns ``\\beta = argmin_{b} \\sum_i w_i(y_i - X_i'b)`` where `X` denotes the matrix of fixed effects `fes`.

### Arguments
* `y` : A `AbstractVector` 
* `fes`: A `Vector{<:FixedEffect}`
* `w`: A vector of weights, i.e. `AbstractWeights`
* `method` : A `Symbol` for the method. Default is :cpu. The option :gpu requires `CuArrays` (in this case, it is recommanded to use the option `double_precision = false`).
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol` : Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `maxiter` : Maximum number of iterations
* `nthreads` : Number of threads


### Returns
* ``\\beta`` : Solution of the least square problem
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
function solve_coefficients!(y::AbstractVector{<: Number}, fes::AbstractVector{<: FixedEffect}, w::AbstractWeights  = Weights(Ones{eltype(y)}(length(y))); 
	method::Symbol = :cpu, double_precision::Bool = eltype(y) == Float64, 
	tol::Real = double_precision ? 1e-8 : 1e-6,  maxiter::Integer = 10000)
	any(ismissing.(fes)) && throw("Some FixedEffect has a missing value for reference or interaction")
	any((length(fe) != length(y) for fe in fes))  && throw("FixedEffects must have the same length as y")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, w, Val{method})
	solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end

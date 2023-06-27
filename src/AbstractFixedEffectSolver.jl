##############################################################################
## 
## AbstractFixedEffectSolver
##
## this type must defined solve_residuals!, solve_coefficients!
##
##############################################################################
abstract type AbstractFixedEffectSolver{T} end
works_with_view(::AbstractFixedEffectSolver) = false

"""
`solve_residuals!(y, fes, w; method = :cpu, double_precision = true, tol = 1e-8, maxiter = 10000, )`

Returns ``y_i - X_i'\\beta`` where ``\\beta = argmin_{b} \\sum_i y_i - X_i'b``, where `X` denotes the matrix of fixed effects `fes`.

### Arguments
* `y` : A `AbstractVector` or A `AbstractMatrix`
* `fes`: A `Vector{<:FixedEffect}`
* `w`: A vector of weights, i.e. `AbstractWeights`
* `method` : A `Symbol` for the method. Default is :cpu. The option :gpu requires `using CUDA` or `using Metal' (in this case, it is recommanded to use the option `double_precision = false`).
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
function solve_residuals!(y::Union{AbstractVector{<: Number}, AbstractMatrix{<: Number}}, fes::AbstractVector{<: FixedEffect}, w::AbstractWeights = uweights(eltype(y), size(y, 1)); 
	method::Symbol = :cpu, double_precision::Bool = eltype(y) == Float64, 
	tol::Real = double_precision ? 1e-8 : 1e-6, maxiter::Integer = 10000,
	nthreads = method == :cpu ? Threads.nthreads() : 256)
	any((length(fe) != size(y, 1) for fe in fes)) && throw("FixedEffects must have the same length as y")
	any(ismissing.(fes)) && throw("FixedEffects must not have missing values")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, w, Val{method}, nthreads)
	solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
end

function solve_residuals!(X::AbstractMatrix, feM::AbstractFixedEffectSolver; progress_bar = true, kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    bar = MiniProgressBar(header = "Demean Variables:", color = Base.info_color(), percentage = false, max = size(X, 2))
    for j in 1:size(X, 2)
    	v0 = time()
        _, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
        v1 = time()
        # remove progress_bar if estimated time lower than 2sec
	    if progress_bar && (j == 1) && ((v1 - v0) * size(X, 2) <= 2)
	    	progress_bar = false
	    end
    	if progress_bar
    		bar.current = j
    	    showprogress(stdout, bar)
    	end
        push!(iterations, iteration)
        push!(convergeds, converged)
    end
    if progress_bar
    	end_progress(stdout, bar)
    end
    return X, iterations, convergeds
end

function solve_residuals!(r::AbstractVector, feM::AbstractFixedEffectSolver{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	# One cannot copy view of Vector (r) on GPU, so first collect the vector
	if works_with_view(feM)
		copyto!(feM.r, r)
	else
		copyto!(feM.tmp, r)
		copyto!(feM.r, feM.tmp)
	end
	if !(feM.weights isa UnitWeights)
		 feM.r .*= sqrt.(feM.weights)
	end
	copyto!(feM.b, feM.r)
	mul!(feM.x, feM.m', feM.b, 1, 0)

	iter, converged = 1, true
    if length(feM.x.x) > 1
        x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
        iter, converged = ch.mvps + 1, ch.isconverged
    end
	mul!(feM.r, feM.m, feM.x, -1, 1)
	if !(feM.weights isa UnitWeights)
		feM.r ./=  sqrt.(feM.weights)
	end
	if works_with_view(feM)
		copy!(r, feM.r)
	else
		copyto!(feM.tmp, feM.r)
		copyto!(r, feM.tmp)
	end
	return r, iter, converged
end




"""
Solve a least square problem for a set of FixedEffects

`solve_coefficients!(y, fes, w; method = :cpu, maxiter = 10000, double_precision = true, tol = 1e-8)`

Returns ``\\beta = argmin_{b} \\sum_i w_i(y_i - X_i'b)`` where `X` denotes the matrix of fixed effects `fes`.

### Arguments
* `y` : A `AbstractVector` 
* `fes`: A `Vector{<:FixedEffect}`
* `w`: A vector of weights, i.e. `AbstractWeights`
* `method` : A `Symbol` for the method. Default is :cpu. The option :gpu requires `using CUDA` (in this case, it is recommanded to use the option `double_precision = false`).
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
function solve_coefficients!(y::AbstractVector{<: Number}, fes::AbstractVector{<: FixedEffect}, w::AbstractWeights = uweights(eltype(y), length(y)); method::Symbol = :cpu, double_precision::Bool = eltype(y) == Float64, 
	tol::Real = double_precision ? 1e-8 : 1e-6,  maxiter::Integer = 10000, 
	nthreads = method == :cpu ? Threads.nthreads() : 256)
	any(ismissing.(fes)) && throw("Some FixedEffect has a missing value for reference or interaction")
	any((length(fe) != length(y) for fe in fes))  && throw("FixedEffects must have the same length as y")
	feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, w, Val{method}, nthreads)
	solve_coefficients!(y, feM; maxiter = maxiter, tol = tol)
end

function FixedEffects.solve_coefficients!(r::AbstractVector, feM::AbstractFixedEffectSolver{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	# One cannot copy view of Vector (r) on GPU, so first collect the vector
	if works_with_view(feM)
		copyto!(feM.b, r)
	else
		copyto!(feM.tmp, r)
		copyto!(feM.b, feM.tmp)
	end
	if !(feM.weights isa UnitWeights)
		feM.b .*= sqrt.(feM.weights)
	end
	fill!(feM.x, 0.0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.scales)
		x .*=  scale
	end
	x = Vector{eltype(r)}[collect(x) for x in feM.x.x]
	full(normalize!(x, feM.m.fes; tol = tol, maxiter = maxiter), feM.m.fes), div(ch.mvps, 2), ch.isconverged
end
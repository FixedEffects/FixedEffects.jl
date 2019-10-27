struct FixedEffectSolverLSMR{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMap{T}
	b::AbstractVector{T}
	r::AbstractVector{T}
	x::FixedEffectCoefficients{T}
	v::FixedEffectCoefficients{T}
	h::FixedEffectCoefficients{T}
	hbar::FixedEffectCoefficients{T}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:lsmr}}) where {T}
	m = FixedEffectLinearMap{T}(fes, weights, Val{:lsmr})
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	x = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	return FixedEffectSolverLSMR(m, b, r, x, v, h, hbar)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverLSMR{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.r, r)
	feM.r .*=  feM.m.sqrtw
	fill!(feM.x, 0)
	copyto!(feM.b, feM.r)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	mul!(feM.r, feM.m, feM.x, -1.0, 1.0)
	feM.r ./=  feM.m.sqrtw
	copyto!(r, feM.r)
	return r, div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolverLSMR; kwargs...)
	iterations = Int[]
	convergeds = Bool[]
	for j in 1:size(X, 2)
		_, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
		push!(iterations, iteration)
		push!(convergeds, converged)
	end
	return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverLSMR{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.b, r)
	feM.b .*=  feM.m.sqrtw
	fill!(feM.x, 0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.colnorm)
		x ./=  scale
	end
	x = Vector{eltype(r)}[x for x in feM.x.x]
	full(normalize!(x, feM.m.fes; tol = tol, maxiter = maxiter), feM.m.fes), div(ch.mvps, 2), ch.isconverged
end


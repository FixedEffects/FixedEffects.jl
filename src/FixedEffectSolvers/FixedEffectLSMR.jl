struct FixedEffectLSMR{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMap{T}
	xs::FixedEffectCoefficients{T}
	v::FixedEffectCoefficients{T}
	h::FixedEffectCoefficients{T}
	hbar::FixedEffectCoefficients{T}
	u::AbstractVector{T}
	r::AbstractVector{T}
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}}) where {T}
	m = FixedEffectLinearMap{T}(fes, sqrtw, Val{:lsmr})
	xs = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	u = zeros(T, length(sqrtw))
	r = zeros(T, length(sqrtw))
	return FixedEffectLSMR(m, xs, v, h, hbar, u, r)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMR; tol::Real = 1e-8, maxiter::Integer = 100_000)
	copyto!(feM.r, r)
	feM.r .*=  feM.m.sqrtw
	copyto!(feM.u, feM.r)
	fill!(feM.xs, 0.0)
	x, ch = lsmr!(feM.xs, feM.m, feM.u, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
	mul!(feM.r, feM.m, feM.xs, -1.0, 1.0)
	feM.r ./=  feM.m.sqrtw
	copyto!(r, feM.r)
	return r, div(ch.mvps, 2), ch.isconverged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMR; tol::Real = 1e-8, maxiter::Integer = 100_000)
	copyto!(feM.u, r)
	feM.u .*= feM.m.sqrtw
	fill!(feM.xs, 0.0)
	x, ch = lsmr!(feM.xs, feM.m, feM.u, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
	for (x, scale) in zip(feM.xs.x, feM.m.scales)
			x .*=  scale
	end
	xs = Vector{eltype(r)}[x for x in feM.xs.x]
	full(normalize!(xs, feM.m.fes; tol = tol, maxiter = maxiter), feM.m.fes), div(ch.mvps, 2), ch.isconverged
end


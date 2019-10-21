##############################################################################
##
## LSMR GPU
##
##############################################################################
using CuArrays.CUDAnative
import CuArrays: allowscalar
allowscalar(false)

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################

# https://github.com/JuliaGPU/CuArrays.jl/issues/306
cuzeros(T::Type, n::Integer) = fill!(CuVector{T}(undef, n), zero(T))
cuones(T::Type, n::Integer) = fill!(CuVector{T}(undef, n), one(T))

function CuArrays.cu(T::Type, fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = cu(T, fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
CuArrays.cu(T::Type, w::Ones) = cuones(T, length(w))
CuArrays.cu(T::Type, w::AbstractVector) = CuVector{T}(w)

##############################################################################
##
## FixedEffectLinearMap on the GPU (code by Paul Schrimpf)
##
##############################################################################

const N_THREADS = 256

function FixedEffectLinearMap{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}}) where {T}
	fes = [cu(T, fe) for fe in fes]
	sqrtw = cu(T, sqrtw)
	colnorm = [colnorm!(cuzeros(T, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes]
	caches = [cache!(cuzeros(T, length(sqrtw)), fe.interaction, sqrtw, scale, fe.refs) for (fe, scale) in zip(fes, colnorm)]
	return FixedEffectLinearMap{T}(fes, sqrtw, colnorm, caches)
end

function colnorm!(fecoef::CuVector, refs::CuVector, y::CuVector, sqrtw::CuVector)
	nblocks = cld(length(refs), N_THREADS) 
	@cuda threads=N_THREADS blocks=nblocks colnorm!_kernel!(fecoef, refs, y, sqrtw)
	fecoef .= sqrt.(fecoef)
end

function colnorm!_kernel!(fecoef, refs, y, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), abs2(y[i] * sqrtw[i]))
	end
end

function cache!(y::CuVector, interaction::CuVector , sqrtw::CuVector, fecoef::CuVector, refs::CuVector)
	nblocks = cld(length(y), N_THREADS) 
	@cuda threads=N_THREADS blocks=nblocks cache_kernel!(y, interaction, sqrtw, fecoef, refs)
	return y
end

function cache_kernel!(y, interaction, sqrtw, fecoef, refs)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += interaction[i] * sqrtw[i] / fecoef[refs[i]]
	end
end

function mean!(fecoef::CuVector, refs::CuVector, α::Number, y::CuVector, cache::CuVector)
	nblocks = cld(length(y), N_THREADS) 
	@cuda threads=N_THREADS blocks=nblocks mean_kernel!(fecoef, refs, α, y, cache)    
end

function mean_kernel!(fecoef, refs, α, y, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), α * y[i] * cache[i])
	end
end

function demean!(y::CuVector, α::Number, fecoef::CuVector, refs::CuVector, cache::CuVector)
	nblocks = cld(length(y), N_THREADS)
	@cuda threads=N_THREADS blocks=nblocks demean_kernel!(y, α, fecoef, refs, cache)
end

function demean_kernel!(y, α, fecoef, refs, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################
struct FixedEffectSolverLSMRGPU{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMap{T}
	b::CuVector{T}
	r::CuVector{T}
	x::FixedEffectCoefficients{T}
	v::FixedEffectCoefficients{T}
	h::FixedEffectCoefficients{T}
	hbar::FixedEffectCoefficients{T}
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
	fes::Vector{<:FixedEffect}
end
	
function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}}) where {T}
	m = FixedEffectLinearMap{T}(fes, sqrtw, Val{:lsmr_gpu})
	b = cuzeros(T, length(sqrtw))
	r = cuzeros(T, length(sqrtw))
	x = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	tmp = zeros(T, length(sqrtw))
	FixedEffectSolverLSMRGPU(m, b, r, x, v, h, hbar, tmp, fes)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverLSMRGPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.tmp, r)
	copyto!(feM.r, feM.tmp)
	feM.r .*=  feM.m.sqrtw
	copyto!(feM.b, feM.r)
	fill!(feM.x, 0.0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	mul!(feM.r, feM.m, feM.x, -1.0, 1.0)
	feM.r ./=  feM.m.sqrtw
	copyto!(feM.tmp, feM.r)
	copyto!(r, feM.tmp)
	return r, div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectSolverLSMRGPU; kwargs...)
	iterations = Int[]
	convergeds = Bool[]
	for j in 1:size(X, 2)
		_, iteration, converged = solve_residuals!(view(x, :, j), feM; kwargs...)
		push!(iterations, iteration)
		push!(convergeds, converged)
	end
	return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverLSMRGPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.tmp, r)
	copyto!(feM.b, feM.tmp)
	feM.b .*= feM.m.sqrtw
	fill!(feM.x, 0.0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.colnorm)
		x ./=  scale
	end
	x = Vector{eltype(r)}[collect(x) for x in feM.x.x]
	full(normalize!(x, feM.fes; tol = tol, maxiter = maxiter), feM.fes), div(ch.mvps, 2), ch.isconverged
end
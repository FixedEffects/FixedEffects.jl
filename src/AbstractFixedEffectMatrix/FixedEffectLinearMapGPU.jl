##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative
import .CuArrays: allowscalar

allowscalar(false)

##############################################################################
##
## Implement special mean! and deman! (code by Paul Schrimpf)
##
##############################################################################

function mean!(fecoef::CuVector, refs::CuVector, y::CuVector, α::Number, cache::CuVector)
    nthreads = 256
    nblocks = div(length(y), nthreads) + 1
    @cuda threads=nthreads blocks=nblocks mean_kernel!(fecoef, refs, y, α, cache)
end

function mean_kernel!(fecoef, refs, y, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(y)
        CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), y[i] * α * cache[i])
    end
    return nothing
end

function demean!(y::CuVector, fecoef::CuVector, refs::CuVector, α::Number, cache::CuVector)
    nthreads = 256
    nblocks = div(length(y), nthreads) + 1
    @cuda threads=nthreads blocks=nblocks demean_kernel!(y, fecoef, refs, α, cache)
end

function demean_kernel!(y, fecoef, refs, α, cache)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds for i = index:stride:length(y)
    	y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
end

function scale!(fecoef::CuVector, refs::CuVector, y::CuVector, sqrtw::CuVector)
	nthreads = 256
	nblocks = div(length(y), nthreads) + 1
	@cuda threads=nthreads blocks=nblocks scale_kernel!(fecoef, refs, y, sqrtw)
	fecoef .= 1.0 ./ sqrt.(fecoef)
end

function scale_kernel!(fecoef, refs, y, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
	    CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), abs2(y[i] * sqrtw[i]))
	end
	return nothing
end

function cache!(y::CuVector, refs::CuVector, interaction::CuVector, fecoef::CuVector, sqrtw::CuVector)
	nthreads = 256
	nblocks = div(length(y), nthreads) + 1
	@cuda threads=nthreads blocks=nblocks cache_kernel!(y, refs, interaction, fecoef, sqrtw)
end

function cache_kernel!(y, refs, interaction, fecoef, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += fecoef[refs[i]] * interaction[i] * sqrtw[i]
	end
	return nothing
end

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################
function CuArrays.cu(T::Type, fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuVector{T}(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
	interaction = collect(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

##############################################################################
##
## Implement AbstractRixedEffectMatrix Interface
##
## GC accounts for large part of timing so important to have temporary arrays
##
##############################################################################

struct FixedEffectLSMRGPU{T} <: AbstractFixedEffectMatrix{T}
	m::FixedEffectLSMR{T}
	tmp::Vector{T} 	# used to convert Abstract{Float64} to Vector{Float32}
	tmp2::CuVector{T} # used to convert Vector{Float32} to CuVector{Float32}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	fes = [cu(Float32, fe) for fe in fes]
	sqrtw = CuVector{Float32}(sqrtw)
	n = length(sqrtw)
	scales = FixedEffectCoefficients([scale!(cuzeros(Float32, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes])
	caches = [cache!(cuzeros(Float32, n), fe.refs, fe.interaction, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
	xs = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes])
	v = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes])
	h = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([cuzeros(Float32, fe.n) for fe in fes])
	u = cuzeros(Float32, n)
	FixedEffectLSMRGPU(FixedEffectLSMR(fes, scales, caches, xs, v, h, hbar, u, sqrtw), zeros(Float32, n), cuzeros(Float32, n))
end
# CuArrays.zero does not give CuVector
cuzeros(T, n::Integer) = fill!(CuVector{T}(undef, n), zero(T))


function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.tmp2, feM.tmp)
	_, iterations, converged = solve_residuals!(feM.tmp2, feM.m; kwargs...)
	copyto!(feM.tmp, feM.tmp2)
	copyto!(r, feM.tmp), iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.tmp2, feM.tmp)
	iterations, converged = _solve_coefficients!(feM.tmp2, feM.m)
	xs = collect.(feM.m.xs.x)
	fes = collect.(feM.m.fes)
	full(normalize!(xs, fes; kwargs...), fes), iterations, converged
end



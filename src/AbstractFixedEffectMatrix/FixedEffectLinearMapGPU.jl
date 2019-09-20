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
        r = refs[i]
        CuArrays.CUDAnative.atomic_add!(pointer(fecoef, r), y[i] * α * cache[i])
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
    for i = index:stride:length(y)
    	@inbounds y[i] += fecoef[refs[i]] * α * cache[i]
    end
    return nothing
end

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################
function CuArrays.cu(fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuVector{Float32}(fe.interaction)
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
##############################################################################
struct FixedEffectLSMRGPU{T} <: AbstractFixedEffectMatrix{T}
	m::FixedEffectLSMR{T}
	tmp::Vector{T} 	# used to convert views, Float64 to Vector{Float32}
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	scales = [_scale!(Vector{Float32}(undef, fe.n), fe, sqrtw) for fe in fes]
	n = length(sqrtw)
	tmp = Vector{Float32}(undef, n)
	caches = [CuVector{Float32}(_cache!(tmp, fe, scale, sqrtw)) for (fe, scale) in zip(fes, scales)]
	scales = cu.(scales)
	fes = cu.(fes)
	sqrtw = CuVector{Float32}(sqrtw)
	xs = FixedEffectCoefficients([CuVector{Float32}(undef, fe.n) for fe in fes])
	v = FixedEffectCoefficients([CuVector{Float32}(undef, fe.n) for fe in fes])
	h = FixedEffectCoefficients([CuVector{Float32}(undef, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([CuVector{Float32}(undef, fe.n) for fe in fes])
	fill!(v, 0.0)
	fill!(h, 0.0)
	fill!(hbar, 0.0)
	u = CuVector{Float32}(undef, n)
	FixedEffectLSMRGPU(FixedEffectLSMR(fes, scales, caches, xs, v, h, hbar, u, sqrtw), tmp)
end


function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	copyto!(feM.tmp, r)
	copyto!(feM.m.u, feM.tmp)
	_, iterations, converged = solve_residuals!(feM.m.u, feM.m; kwargs...)
	copyto!(feM.tmp, feM.m.u)
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



using .CuArrays
using .CuArrays.CUDAnative


##############################################################################
##
## Implement special mean! and deman!
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
## Implement AbstractRixedEffectMatrix Interface
##
##############################################################################
struct FixedEffectLSMRGPU{T} <: AbstractFixedEffectMatrix{T}
	m::FixedEffectLSMR{T}
end


function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	fes = CuArray.(fes)
	sqrtw = CuArray(convert(Vector{Float32}, sqrtw))
	scales = [CuArrays_scale(fe, sqrtw) for fe in fes] 
	caches = [CuArrays_cache(fe, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
	xs = CuArraysFixedEffectCoefficients(fes)
	v = CuArraysFixedEffectCoefficients(fes)
	h = CuArraysFixedEffectCoefficients(fes)
	hbar = CuArraysFixedEffectCoefficients(fes)
	u = CuArrays.zeros(length(first(fes)))
	return FixedEffectLSMR(fes, scales, caches, xs, v, h, hbar, u, sqrtw)
end

# convert FixedEffects between CPU and GPU
function CuArrays.CuArray(fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuArray(convert(Vector{Float32}, fe.interaction))
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

function CuArrays_scale(fe::FixedEffect{<: CuVector}, sqrtw::CuVector)
    out = CuArrays.zeros(fe.n)
    for i in eachindex(fe.refs)
        out[fe.refs[i]] += abs2(fe.interaction[i] * sqrtw[i])
    end
    for i in eachindex(out)
        out[i] = out[i] > 0.0 ? (1.0 / sqrt(out[i])) : 0.0
    end
    return out
end

function CuArrays_cache(fe::FixedEffect{<: CuVector}, scale::CuVector, sqrtw::CuVector)
    out = CuArrays.zeros(length(fe.refs))
    @inbounds @simd for i in eachindex(out)
        out[i] = scale[fe.refs[i]] * fe.interaction[i] * sqrtw[i]
    end
    return out
end

function CuArraysFixedEffectCoefficients(fes::Vector{<:FixedEffect})
    FixedEffectCoefficients([CuArrays.zeros(fe.n) for fe in fes])
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	cur = cu(r)
	cur, iterations, converged = solve_residuals!(cur, feM.m; kwargs...)
	copyto!(r, cur), iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	cur = cu(r)
	iterations, converged = _solve_coefficients!(r, feM.m)
	xs = collect.(feM.m.xs.x)
	fes = collect.(feM.m.fes)
	full(normalize!(xs, fes; kwargs...), fes), iterations, converged
end

function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
	interaction = collect(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
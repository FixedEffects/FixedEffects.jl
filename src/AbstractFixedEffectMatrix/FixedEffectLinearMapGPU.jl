
##############################################################################
##
## LSMR GPU
##
##############################################################################
using .CuArrays
using .CuArrays.CUDAnative

# convert FixedEffects between CPU and GPU
function CuArrays.CuArray(fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = CuArray(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

function Base.collect(fe::FixedEffect{<: CuVector})
	refs = collect(fe.refs)
	interaction = collect(fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end

# convert FixedEffectCoefficient between CPU and GPU
CuArrays.CuArray(x::FixedEffectCoefficients) = FixedEffectCoefficients(CuArray.(x.x))
Base.collect(x::FixedEffectCoefficients{<: CuVector}) = FixedEffectCoefficients(collect.(x))

# convert FixedEffectLSMR between CPU and GPU
function CuArrays.CuArray(m::FixedEffectLSMR)
	FixedEffectLSMR(CuArray.(m.fes), CuArray.(m.scales), CuArray.(m.caches), CuArray(m.xs), CuArray(m.v), CuArray(m.h), CuArray(m.hbar), CuArray(m.u), CuArray(m.sqrtw))
end

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
struct FixedEffectLSMRGPU <: AbstractFixedEffectMatrix
	m::FixedEffectLSMR
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_gpu}})
	FixedEffectLSMRGPU(CuArray(FixedEffectMatrix(fes, sqrtw, Val{:lsmr})))
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	cur = CuArray(r)
	cur, iterations, converged = solve_residuals!(cur, feM.m; kwargs...)
	copyto!(r, cur), iterations, converged
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRGPU; kwargs...)
	cur = CuArray(r)
	feM = feM.m
	cur .*= feM.sqrtw
	iterations, converged = solve!(feM, r; kwargs...)
	for (x, scale) in zip(feM.xs, feM.scales)
	   x .*=  scale
	end 
	x_gpu = collect(feM.xs.x)
	fes_gpu = collect(feM.fes)
	full(normalize!(x_gpu, fes_gpu; kwargs...), fes_gpu), iterations, converged
end
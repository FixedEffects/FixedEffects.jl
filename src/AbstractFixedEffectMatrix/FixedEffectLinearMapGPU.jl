
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
CUDAnative.InvalidIRError(CUDAnative.CompilerJob(FixedEffects.demean_kernel!, Tuple{CUDAnative.CuDeviceArray{Float64,1,CUDAnative.AS.Global},CUDAnative.CuDeviceArray{Float64,1,CUDAnative.AS.Global},CUDAnative.CuDeviceArray{UInt32,1,CUDAnative.AS.Global},Int64,CUDAnative.CuDeviceArray{Float64,1,CUDAnative.AS.Global}}, v"3.7.0", true, nothing, nothing, nothing, nothing), Tuple{String,Array{Base.StackTraces.StackFrame,1},Any}[("call to the Julia runtime", [demean_kernel! at FixedEffectLinearMapGPU.jl:61], "jl_f_getfield"), ("dynamic function invocation", [demean_kernel! at FixedEffectLinearMapGPU.jl:62], iterate), ("dynamic function invocation", [* at operators.jl:502, demean_kernel! at FixedEffectLinearMapGPU.jl:62], iterate), ("use of an undefined name", [demean_kernel! at FixedEffectLinearMapGPU.jl:59], :blockIdx), ("dynamic function invocation", [demean_kernel! at FixedEffectLinearMapGPU.jl:59], iterate), ("use of an undefined name", [demean_kernel! at FixedEffectLinearMapGPU.jl:59], :blockDim), ("use of an undefined name", [demean_kernel! at FixedEffectLinearMapGPU.jl:59], :threadIdx), ("use of an undefined name", [demean_kernel! at FixedEffectLinearMapGPU.jl:60], :blockDim), ("dynamic function invocation", [demean_kernel! at FixedEffectLinearMapGPU.jl:60], iterate), ("use of an undefined name", [demean_kernel! at FixedEffectLinearMapGPU.jl:60], :gridDim), ("dynamic function invocation", [demean_kernel! at FixedEffectLinearMapGPU.jl:61], iterate)])

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

function solve_residuals!(r::AbstractVector{Float64}, feM::FixedEffectLSMRGPU; kwargs...)
	cur = CuArray(r)
	cur, iterations, converged = solve_residuals!(cur, feM.m; kwargs...)
	copy!(r, cur), iterations, converged
end

function solve_coefficients!(r::AbstractVector{Float64}, feM::FixedEffectLSMRGPU; kwargs...)
	cur = CuArray(r)
	cur .*= feM.sqrtw
	iterations, converged = solve!(feM, r; kwargs...)
	for (x, scale) in zip(feM.xs, feM.scales)
	   x .*=  scale
	end 
	x = collect(feM.xs.x)
	fes = collect(feM.fes)
	full(normalize!(x, fes; kwargs...), fes), iterations, converged
end
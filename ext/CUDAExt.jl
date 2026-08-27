module CUDAExt
using FixedEffects, CUDA
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!, AbstractFixedEffectLinearMap, copy_internal!, AbsorptionPlan, AbsorbedBlock, block_width
CUDA.allowscalar(false)

##############################################################################
##
## CUDA backend — same layout as src/CPU.jl and ext/MetalExt.jl:
##   1. FixedEffectLinearMapCUDA: plan transfer, mul!, kernels;
##   2. FixedEffectSolverCUDA: solver storage and interface.
## The AbsorptionPlan (block transforms and whitened row values) is built on
## the CPU; refs and qrows are moved to the device and consumed by fused
## block kernels.
##
##############################################################################

##############################################################################
##
## 1. FixedEffectLinearMapCUDA
##
##############################################################################

## 1a) FixedEffectLinearMapCUDA Constructor

_cu(T::Type, w::UnitWeights) = fill!(CuVector{T}(undef, length(w)), w[1])
_cu(T::Type, w::AbstractVector) = CuVector{T}(convert(Vector{T}, w))

mutable struct FixedEffectLinearMapCUDA{T,P<:AbsorptionPlan} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	plan::P
end

function FixedEffectLinearMapCUDA{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights) where {T}
	plan = _cu_plan(T, fes, weights)
	return FixedEffectLinearMapCUDA{T,typeof(plan)}(fes, plan)
end

function _cu_plan(::Type{T}, fes::Vector{<:FixedEffect}, weights::AbstractWeights) where {T}
	cpu_plan = AbsorptionPlan(T, fes, weights)
	blocks = [AbsorbedBlock(CuArray(block.refs), block.interactions, block.n, block.input_terms)
		for block in cpu_plan.blocks]
	qrows = [CuArray(q) for q in cpu_plan.qrows]
	return AbsorptionPlan(blocks, cpu_plan.transforms, cpu_plan.ranks, qrows)
end

## 1b) FixedEffectLinearMapCUDA mul!

## Implement right multiplication
function LinearAlgebra.mul!(y::CuVector, fem::FixedEffectLinearMapCUDA{T},
		fecoefs::FixedEffectCoefficients, α::Number, β::Number) where {T}
	if iszero(β)
		fill!(y, zero(T))
		β = one(β)
	end
	for (coef_block, block, qrows) in zip(fecoefs.x, fem.plan.blocks, fem.plan.qrows)
		_scatter_block!(y, block.refs, qrows, coef_block, α, β)
		β = one(β)
	end
	return y
end

function _scatter_block!(y::CuVector, refs::CuVector, qrows::CuMatrix,
		coef_block::CuMatrix, α::Number, β::Number)
	nthreads = 256
	nblocks = cld(length(y), nthreads)
	@cuda threads=nthreads blocks=nblocks scatter_block_kernel!(y, refs, qrows, coef_block, α, β, size(coef_block, 1))
	return y
end

function scatter_block_kernel!(y, refs, qrows, coef_block, α, β, k)
	index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	i = index
	@inbounds while i <= length(y)
		g = refs[i]
		fit = zero(eltype(y))
		for c in 1:k
			fit += coef_block[c, g] * qrows[c, i]
		end
		y[i] = β * y[i] + α * fit
		i += stride
	end
	return nothing
end

## 1c) FixedEffectLinearMapCUDA mul!, Adjoint

## Implement left multiplication
function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients,
		Cfem::Adjoint{T, <:FixedEffectLinearMapCUDA{T}},
		y::CuVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (coef_block, block, qrows) in zip(fecoefs.x, fem.plan.blocks, fem.plan.qrows)
		_gather_block!(coef_block, block.refs, qrows, y, α)
	end
	return fecoefs
end

function _gather_block!(coef_block::CuMatrix, refs::CuVector, qrows::CuMatrix,
		y::CuVector, α::Number)
	nthreads = 256
	nblocks = cld(length(y), nthreads)
	@cuda threads=nthreads blocks=nblocks gather_block_kernel!(coef_block, refs, qrows, y, α, size(coef_block, 1))
	return coef_block
end

function gather_block_kernel!(coef_block, refs, qrows, y, α, k)
	index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	i = index
	@inbounds while i <= length(y)
		g = refs[i]
		yi = α * y[i]
		for c in 1:k
			CUDA.@atomic coef_block[c, g] += yi * qrows[c, i]
		end
		i += stride
	end
	return nothing
end

##############################################################################
##
## 2. FixedEffectSolverCUDA
##
##############################################################################

mutable struct FixedEffectSolverCUDA{T} <: FixedEffects.AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapCUDA{T}
	weights::CuVector{T}
	b::CuVector{T}
	r::CuVector{T}
	x::FixedEffectCoefficients
	v::FixedEffectCoefficients
	h::FixedEffectCoefficients
	hbar::FixedEffectCoefficients
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
end

function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:CUDA}}) where {T}
	m = FixedEffectLinearMapCUDA{T}(fes, weights)
	b = CUDA.zeros(T, length(weights))
	r = CUDA.zeros(T, length(weights))
	x = FixedEffectCoefficients([CUDA.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	v = FixedEffectCoefficients([CUDA.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	h = FixedEffectCoefficients([CUDA.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	hbar = FixedEffectCoefficients([CUDA.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	tmp = zeros(T, length(weights))
	return FixedEffectSolverCUDA{T}(m, _cu(T, weights), b, r, x, v, h, hbar, tmp)
end

function FixedEffects.update_weights!(feM::FixedEffectSolverCUDA{T}, weights::AbstractWeights) where {T}
	copyto!(feM.weights, _cu(T, weights))
	feM.m.plan = _cu_plan(T, feM.m.fes, weights)
	return feM
end

function FixedEffects.copy_internal!(feM::FixedEffectSolverCUDA, field::Symbol, r::AbstractVector)
	copyto!(feM.tmp, r)
	copyto!(getfield(feM, field), feM.tmp)
end

function FixedEffects.copy_internal!(r::AbstractVector, feM::FixedEffectSolverCUDA, field::Symbol)
	copyto!(feM.tmp, getfield(feM, field))
	copyto!(r, feM.tmp)
end


end

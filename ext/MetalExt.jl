module MetalExt
using FixedEffects, Metal
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!, AbstractFixedEffectLinearMap, copy_internal!, AbsorptionPlan, AbsorbedBlock, _group_permutation, block_width
Metal.allowscalar(false)

##############################################################################
##
## Metal backend — same layout as src/CPU.jl and ext/CUDAExt.jl:
##   1. FixedEffectLinearMapMetal: gather strategies, plan transfer, mul!, kernels;
##   2. FixedEffectSolverMetal: solver storage and interface.
## The AbsorptionPlan (block transforms and whitened row values) is built on
## the CPU; refs and qrows are moved to the device and consumed by fused
## block kernels.
##
##############################################################################

##############################################################################
##
## 1. FixedEffectLinearMapMetal
##
##############################################################################

## 1a) FixedEffectLinearMapMetal Constructor

_mtl(T::Type, w::UnitWeights) = Metal.ones(T, length(w))
_mtl(T::Type, w::AbstractVector) = MtlVector{T}(convert(Vector{T}, w))

function _metal_threadgroup_width()
	width = Int(device().maxThreadsPerThreadgroup.width)
	return prevpow(2, width)
end

# Per-block plan for the adjoint gather (A'u), chosen once at construction:
# bucketize (one threadgroup per group) for low cardinality, else atomic adds.
struct AtomicGather end
struct BucketGather{V<:AbstractVector}
	perm::V
	offsets::V
end

mutable struct FixedEffectLinearMapMetal{T,P<:AbsorptionPlan} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	plan::P
	gathers::Vector{Union{AtomicGather, BucketGather}}
end

function FixedEffectLinearMapMetal{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights) where {T}
	plan = _mtl_plan(T, fes, weights)
	G = Union{AtomicGather, BucketGather}
	gathers = Vector{G}(undef, length(plan.blocks))
	for i in eachindex(plan.blocks)
		refs = fes[plan.blocks[i].input_terms[1]].refs
		gathers[i] = _gather_strategy(refs, plan.blocks[i].n)
	end
	return FixedEffectLinearMapMetal{T,typeof(plan)}(fes, plan, gathers)
end

function _gather_strategy(refs::AbstractVector{<:Integer}, nlevels::Int)
	if nlevels < min(100_000, div(length(refs), 16))
		_, offsets, perm = _group_permutation(refs, nlevels)
		return BucketGather(MtlVector{Int}(perm), MtlVector{Int}(offsets))
	else
		return AtomicGather()
	end
end

function _mtl_plan(::Type{T}, fes::Vector{<:FixedEffect}, weights::AbstractWeights) where {T}
	cpu_plan = AbsorptionPlan(T, fes, weights)
	blocks = [AbsorbedBlock(MtlArray(block.refs), block.interactions, block.n, block.input_terms)
		for block in cpu_plan.blocks]
	qrows = [MtlArray(q) for q in cpu_plan.qrows]
	return AbsorptionPlan(blocks, cpu_plan.transforms, cpu_plan.ranks, qrows)
end

## 1b) FixedEffectLinearMapMetal mul!

## Implement right multiplication
function LinearAlgebra.mul!(y::MtlVector, fem::FixedEffectLinearMapMetal{T},
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

function _scatter_block!(y::MtlVector, refs::MtlVector, qrows::MtlMatrix,
		coef_block::MtlMatrix, α::Number, β::Number)
	nthreads = _metal_threadgroup_width()
	nblocks = cld(length(y), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks scatter_block_kernel!(y, refs, qrows, coef_block, α, β, size(coef_block, 1))
	return y
end

function scatter_block_kernel!(y, refs, qrows, coef_block, α, β, k)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		@inbounds begin
			g = refs[i]
			fit = zero(eltype(y))
			for c in 1:k
				fit += coef_block[c, g] * qrows[c, i]
			end
			y[i] = β * y[i] + α * fit
		end
	end
	return nothing
end

## 1c) FixedEffectLinearMapMetal mul!, Adjoint

## Implement left multiplication
function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients,
		Cfem::Adjoint{T, <:FixedEffectLinearMapMetal{T}},
		y::MtlVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (coef_block, block, qrows, gather) in zip(fecoefs.x, fem.plan.blocks, fem.plan.qrows, fem.gathers)
		_gather_block!(coef_block, block.refs, qrows, y, α, gather)
	end
	return fecoefs
end

function _gather_block!(coef_block::MtlMatrix, refs::MtlVector, qrows::MtlMatrix,
		y::MtlVector, α::Number, gather::BucketGather)
	n = size(coef_block, 2)
	nthreads = _metal_threadgroup_width()
	Metal.@sync @metal threads=nthreads groups=n gather_block_kernel_bin!(coef_block, α, y, qrows, gather.perm, gather.offsets, Val(nthreads), size(coef_block, 1))
	return coef_block
end

function _gather_block!(coef_block::MtlMatrix, refs::MtlVector, qrows::MtlMatrix,
		y::MtlVector, α::Number, ::AtomicGather)
	nthreads = _metal_threadgroup_width()
	nblocks = cld(length(y), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks gather_block_kernel!(coef_block, refs, α, y, qrows, size(coef_block, 1))
	return coef_block
end

function gather_block_kernel_bin!(coef_block, α, y, qrows, perm, offsets, ::Val{NT}, k) where {NT}
	g = Int(threadgroup_position_in_grid().x)
	tid = Int(thread_position_in_threadgroup().x)
	nt = Int(threads_per_threadgroup().x)
	T = eltype(coef_block)
	shared = Metal.MtlThreadGroupArray(T, NT)

	start = @inbounds offsets[g]
	stop = @inbounds offsets[g + 1] - 1

	for c in 1:k
		acc = zero(T)
		j = start + tid - 1
		while j <= stop
			i = @inbounds perm[j]
			@inbounds acc += α * y[i] * qrows[c, i]
			j += nt
		end

		@inbounds shared[tid] = acc
		Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

		offset = nt ÷ 2
		while offset > 0
			if tid <= offset
				@inbounds shared[tid] += shared[tid + offset]
			end
			Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
			offset ÷= 2
		end

		if tid == 1
			@inbounds coef_block[c, g] += shared[1]
		end
	end

	return nothing
end

function gather_block_kernel!(coef_block, refs, α, y, qrows, k)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		@inbounds begin
			g = refs[i]
			yi = α * y[i]
			for c in 1:k
				idx = c + (g - 1) * k
				Metal.atomic_fetch_add_explicit(pointer(coef_block, idx), yi * qrows[c, i])
			end
		end
	end
	return nothing
end

##############################################################################
##
## 2. FixedEffectSolverMetal
##
##############################################################################

mutable struct FixedEffectSolverMetal{T} <: FixedEffects.AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapMetal{T}
	weights::MtlVector{T}
	b::MtlVector{T}
	r::MtlVector{T}
	x::FixedEffectCoefficients
	v::FixedEffectCoefficients
	h::FixedEffectCoefficients
	hbar::FixedEffectCoefficients
	tmp::Vector{T}
end


function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:Metal}}) where {T}
	T === Float32 || throw(ArgumentError("The Metal backend supports Float32 solves only; pass double_precision=false or use method=:cpu for Float64."))
	m = FixedEffectLinearMapMetal{T}(fes, weights)
	b = Metal.zeros(T, length(weights))
	r = Metal.zeros(T, length(weights))
	x = FixedEffectCoefficients([Metal.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	v = FixedEffectCoefficients([Metal.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	h = FixedEffectCoefficients([Metal.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	hbar = FixedEffectCoefficients([Metal.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	tmp = zeros(T, length(weights))
	return FixedEffectSolverMetal{T}(m, _mtl(T, weights), b, r, x, v, h, hbar, tmp)
end


function FixedEffects.update_weights!(feM::FixedEffectSolverMetal{T}, weights::AbstractWeights) where {T}
	copyto!(feM.weights, _mtl(T, weights))
	feM.m.plan = _mtl_plan(T, feM.m.fes, weights)
	return feM
end

function FixedEffects.recover_coefficients(feM::FixedEffectSolverMetal{T}, ::Type{Tout}) where {T,Tout}
	synchronize()
	return FixedEffects.recover_coefficients(T, feM.m.fes, feM.m.plan, Matrix{T}[Array(x) for x in feM.x.x], Tout)
end

function FixedEffects.copy_internal!(feM::FixedEffectSolverMetal{T}, field::Symbol, r::AbstractVector) where {T}
	synchronize()
	copyto!(feM.tmp, r)
	copyto!(getfield(feM, field), feM.tmp)
end

function FixedEffects.copy_internal!(r::AbstractVector, feM::FixedEffectSolverMetal{T}, field::Symbol) where {T}
	synchronize()
	copyto!(feM.tmp, getfield(feM, field))
	copyto!(r, feM.tmp)
end


end

module CUDAExt
using FixedEffects, CUDA
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!, AbstractFixedEffectLinearMap, copy_internal!, AbsorptionPlan, AbsorbedBlock, _group_permutation, block_width
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

# Per-block plan for the adjoint gather (A'u), chosen once at construction:
# bucketize (one thread block per group) for low cardinality, else atomic adds.
struct AtomicGather end
struct BucketGather{V<:AbstractVector}
	perm::V
	offsets::V
end

mutable struct FixedEffectLinearMapCUDA{T,P<:AbsorptionPlan} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	plan::P
	gathers::Vector{Union{AtomicGather, BucketGather}}
end

function FixedEffectLinearMapCUDA{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights) where {T}
	plan = _cu_plan(T, fes, weights)
	G = Union{AtomicGather, BucketGather}
	gathers = Vector{G}(undef, length(plan.blocks))
	for i in eachindex(plan.blocks)
		refs = fes[plan.blocks[i].input_terms[1]].refs
		gathers[i] = _gather_strategy(refs, plan.blocks[i].n)
	end
	return FixedEffectLinearMapCUDA{T,typeof(plan)}(fes, plan, gathers)
end

function _gather_strategy(refs::AbstractVector{<:Integer}, nlevels::Int)
	if nlevels < min(100_000, div(length(refs), 16))
		_, offsets, perm = _group_permutation(refs, nlevels)
		return BucketGather(CuVector{Int}(perm), CuVector{Int}(offsets))
	else
		return AtomicGather()
	end
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
	for (coef_block, block, qrows, gather) in zip(fecoefs.x, fem.plan.blocks, fem.plan.qrows, fem.gathers)
		_gather_block!(coef_block, block.refs, qrows, y, α, gather)
	end
	return fecoefs
end

function _gather_block!(coef_block::CuMatrix, refs::CuVector, qrows::CuMatrix,
		y::CuVector, α::Number, gather::BucketGather)
	nthreads = 256
	nblocks = size(coef_block, 2)
	@cuda threads=nthreads blocks=nblocks gather_block_kernel_bin!(coef_block, α, y, qrows,
		gather.perm, gather.offsets, Val(nthreads), size(coef_block, 1))
	return coef_block
end

function _gather_block!(coef_block::CuMatrix, refs::CuVector, qrows::CuMatrix,
		y::CuVector, α::Number, ::AtomicGather)
	nthreads = 256
	nblocks = cld(length(y), nthreads)
	@cuda threads=nthreads blocks=nblocks gather_block_kernel!(coef_block, refs, qrows, y, α, size(coef_block, 1))
	return coef_block
end

function gather_block_kernel_bin!(coef_block, α, y, qrows, perm, offsets,
		::Val{NT}, k) where {NT}
	g = Int(blockIdx().x)
	tid = Int(threadIdx().x)
	T = eltype(coef_block)
	shared = CUDA.CuStaticSharedArray(T, NT)
	start = @inbounds offsets[g]
	stop = @inbounds offsets[g + 1] - 1

	for c in 1:k
		acc = zero(T)
		j = start + tid - 1
		while j <= stop
			i = @inbounds perm[j]
			@inbounds acc += α * y[i] * qrows[c, i]
			j += NT
		end

		@inbounds shared[tid] = acc
		CUDA.sync_threads()
		offset = NT ÷ 2
		while offset > 0
			if tid <= offset
				@inbounds shared[tid] += shared[tid + offset]
			end
			CUDA.sync_threads()
			offset ÷= 2
		end
		if tid == 1
			@inbounds coef_block[c, g] += shared[1]
		end
	end
	return nothing
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
## 1d) Fused bidiagonalization step (single-pass LSMR iterations)
##
## One kernel computes u ← Σ_blocks A_b v_b + c u and accumulates ‖u‖² on the
## fly: block-reduced in the working precision, then one Float64 atomic per
## thread block so the cross-block sum does not drift (see the _norm2 note in
## src/utils/lsmr.jl). This replaces one scatter launch per block plus a
## separate norm reduction; the gathers then run through their per-block
## strategies on the raw u. Blocks are passed as tuples of device arrays so
## the kernel unrolls across them.
##
##############################################################################

function FixedEffects.bidiag_forward!(u::CuVector{T}, g::FixedEffectCoefficients,
		fem::FixedEffectLinearMapCUDA{T}, v::FixedEffectCoefficients, c::Number) where {T}
	blocks = Tuple(fem.plan.blocks)
	refss = map(block -> block.refs, blocks)
	qrowss = Tuple(fem.plan.qrows)
	vs = Tuple(v.x)
	normacc = CUDA.zeros(Float64, 1)
	nthreads = 256
	nblocks = cld(length(u), nthreads)
	@cuda threads=nthreads blocks=nblocks bidiag_scatter_kernel!(u, refss, qrowss, vs, T(c),
		normacc, Val(nthreads))
	fill!(g, zero(T))
	for (coef_block, block, qrows, gather) in zip(g.x, fem.plan.blocks, fem.plan.qrows, fem.gathers)
		_gather_block!(coef_block, block.refs, qrows, u, one(T), gather)
	end
	return T(sqrt(Array(normacc)[1]))
end

function bidiag_scatter_kernel!(u, refss, qrowss, vs, c, normacc, ::Val{NT}) where {NT}
	T = eltype(u)
	tid = Int(threadIdx().x)
	shared = CUDA.CuStaticSharedArray(T, NT)
	index = (Int(blockIdx().x) - 1) * NT + tid
	stride = NT * Int(gridDim().x)
	acc = zero(T)
	i = index
	@inbounds while i <= length(u)
		ui = c * u[i] + _device_fit(refss, qrowss, vs, i)
		u[i] = ui
		acc += ui * ui
		i += stride
	end
	@inbounds shared[tid] = acc
	CUDA.sync_threads()
	offset = NT ÷ 2
	while offset > 0
		if tid <= offset
			@inbounds shared[tid] += shared[tid + offset]
		end
		CUDA.sync_threads()
		offset ÷= 2
	end
	if tid == 1
		CUDA.@atomic normacc[1] += Float64(shared[1])
	end
	return nothing
end

# Recursion over the block tuples, as in the CPU kernels.
@inline _device_fit(::Tuple{}, ::Tuple{}, ::Tuple{}, i) = false
@inline function _device_fit(refss::Tuple, qrowss::Tuple, vs::Tuple, i)
	refs = first(refss)
	qrows = first(qrowss)
	vcoef = first(vs)
	@inbounds gr = refs[i]
	fit = zero(eltype(qrows))
	for col in 1:size(qrows, 1)
		@inbounds fit += vcoef[col, gr] * qrows[col, i]
	end
	return fit + _device_fit(Base.tail(refss), Base.tail(qrowss), Base.tail(vs), i)
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
	g::FixedEffectCoefficients
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
	g = FixedEffectCoefficients([CUDA.zeros(T, block_width(block), block.n) for block in m.plan.blocks])
	tmp = zeros(T, length(weights))
	return FixedEffectSolverCUDA{T}(m, _cu(T, weights), b, r, x, v, h, hbar, g, tmp)
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

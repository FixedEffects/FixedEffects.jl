##############################################################################
##
## CPU backend
##
## Same layout as ext/CUDAExt.jl and ext/MetalExt.jl:
##   1. FixedEffectLinearMapCPU — the whitened operator over the
##      AbsorptionPlan: gather strategies, gather/scatter kernels, and mul!
##      for the map and its adjoint (all that lsmr! needs);
##   2. FixedEffectSolverCPU — solver storage and interface.
##
##############################################################################

##############################################################################
##
## 1. FixedEffectLinearMapCPU 
##
##############################################################################



## 1a) FixedEffectLinearMapCPU  Constructor



# Per-block plan for the adjoint gather (A'u), chosen once at construction and
# dispatched on by gather_block!.
struct SerialGather end
struct ThreadedGather{M<:AbstractMatrix}
	buffers::Vector{M}              # one k × nlevels accumulator per thread
	ranges::Vector{UnitRange{Int}}  # contiguous row chunks
end

mutable struct FixedEffectLinearMapCPU{T,F<:Vector{<:FixedEffect},P<:AbsorptionPlan,G<:AbstractVector} <: AbstractFixedEffectLinearMap{T}
	fes::F
	plan::P
	gathers::G
end

# The struct definitions must precede this constructor (a constructor method
# signature is evaluated at definition time, unlike ordinary function calls).
function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect},
	weights::AbstractVector = uweights(T, length(fes[1].refs))) where {T}
	plan = AbsorptionPlan(T, fes, weights)
	N = length(fes[1].refs)
	nt = nthreads()
	ranges = _row_chunks(N, nt)
	G = Union{SerialGather, ThreadedGather{Matrix{T}}}
	gathers = G[_gather_strategy(T, block, N, nt, ranges) for block in plan.blocks]
	return FixedEffectLinearMapCPU{T,typeof(fes),typeof(plan),typeof(gathers)}(fes, plan, gathers)
end

# Toggle to force the serial baseline (e.g. for benchmarking); threading is on by default.
const _USE_THREADED_GATHER = Ref(true)
# Threading the gather pays off only when the nt per-thread accumulators of size
# k × nlevels fit in cache; beyond that the fill/merge memory traffic dominates
# and serial is faster.
const _GATHER_BUFFER_BUDGET = 8 * 1024 * 1024   # bytes
const _GATHER_MIN_ROWS = 100_000                # below this, threading overhead isn't worth it

# Per block, thread the gather only if the accumulators fit in cache and N is large.
function _gather_strategy(::Type{T}, block::AbsorbedBlock, N::Int, nt::Int,
		ranges::Vector{UnitRange{Int}}) where {T}
	k = block_width(block)
	if _USE_THREADED_GATHER[] && nt > 1 && N >= _GATHER_MIN_ROWS &&
			nt * k * block.nlevels * sizeof(T) <= _GATHER_BUFFER_BUDGET
		return ThreadedGather([zeros(T, k, block.nlevels) for _ in 1:nt], ranges)
	else
		return SerialGather()
	end
end


## 1b) FixedEffectLinearMapCPU mul!

## Implement right multiplication
function LinearAlgebra.mul!(y::AbstractVector, fem::FixedEffectLinearMapCPU{T},
		fecoefs::FixedEffectCoefficients, α::Number, β::Number) where {T}
	# β applies once, fused into the first scatter; later blocks accumulate
	for (coef_block, block, qrows) in zip(fecoefs.x, fem.plan.blocks, fem.plan.qrows)
		scatter_block!(y, block, coef_block, qrows, α, β)
		β = one(β)
	end
	return y
end

# y[i] += α * sum over c of coef_block[c, refs[i]] * qrows[c, i], the forward map A x.
function scatter_block!(y::AbstractVector, block::AbsorbedBlock, coef_block::AbstractMatrix,
		qrows::AbstractMatrix{T},
		α::Number = one(T)) where {T}
	@spawn_for_chunks 100_000 for i in eachindex(y)
		@inbounds begin
			g = block.refs[i]
			fit = zero(T)
			for c in 1:block_width(block)
				fit += coef_block[c, g] * qrows[c, i]
			end
			y[i] += α * fit
		end
	end
	return y
end

# Fused y = β * y + α * (A x) so mul! avoids a separate scaling pass over y.
function scatter_block!(y::AbstractVector, block::AbsorbedBlock, coef_block::AbstractMatrix,
		qrows::AbstractMatrix{T}, α::Number, β::Number) where {T}
	if isone(β)
		return scatter_block!(y, block, coef_block, qrows, α)
	end
	if iszero(β)
		@spawn_for_chunks 100_000 for i in eachindex(y)
			@inbounds begin
				g = block.refs[i]
				fit = zero(T)
				for c in 1:block_width(block)
					fit += coef_block[c, g] * qrows[c, i]
				end
				y[i] = α * fit
			end
		end
	else
		@spawn_for_chunks 100_000 for i in eachindex(y)
			@inbounds begin
				g = block.refs[i]
				fit = zero(T)
				for c in 1:block_width(block)
					fit += coef_block[c, g] * qrows[c, i]
				end
				y[i] = β * y[i] + α * fit
			end
		end
	end
	return y
end

## 1c) FixedEffectLinearMapCPU mul!, Adjoint


## Implement left multiplication
function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients,
	Cfem::Adjoint{T, <:FixedEffectLinearMapCPU{T}},
	y::AbstractVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (coef_block, block, qrows, gather) in zip(fecoefs.x, fem.plan.blocks, fem.plan.qrows, fem.gathers)
		gather_block!(coef_block, block, qrows, y, α, gather)
	end
	return fecoefs
end


# Serial: one pass over all rows straight into coef_block (which already holds β * old).
gather_block!(coef_block::AbstractMatrix, block::AbsorbedBlock, qrows::AbstractMatrix,
	y::AbstractVector, α::Number, ::SerialGather) =
	_gather_block!(coef_block, block, qrows, y, α, eachindex(y))

# Threaded: each thread reduces its row chunk into a private (cache-resident) buffer,
# then the buffers are summed into coef_block.
function gather_block!(coef_block::AbstractMatrix, block::AbsorbedBlock, qrows::AbstractMatrix,
	y::AbstractVector, α::Number, g::ThreadedGather)
	@threads for t in eachindex(g.buffers)
		buf = g.buffers[t]
		fill!(buf, zero(eltype(buf)))
		_gather_block!(buf, block, qrows, y, α, g.ranges[t])
	end
	@inbounds for buf in g.buffers
		@simd for j in eachindex(coef_block)
			coef_block[j] += buf[j]
		end
	end
	return coef_block
end


# Kernels. block_width is the length of the interactions tuple, so it is a
# compile-time constant: the inner loops over columns unroll separately for
# each block width, and no hand-written k = 1 or k = 2 specializations are
# needed. Inside the @spawn_for_chunks closures the loop bound must be written
# block_width(block) (not a captured integer) for the constant to survive.

# coef_block[c, refs[i]] += α * y[i] * qrows[c, i] over one row range.
# No @simd: distinct i may write the same coef_block column.
function _gather_block!(coef_block::AbstractMatrix, block::AbsorbedBlock,
		qrows::AbstractMatrix{T}, y::AbstractVector, α::Number, range) where {T}
	k = block_width(block)
	@fastmath @inbounds for i in range
		g = block.refs[i]
		yi = α * y[i]
		for c in 1:k
			coef_block[c, g] += yi * qrows[c, i]
		end
	end
	return coef_block
end

##############################################################################
##
## 2. FixedEffectSolverCPU
##
##############################################################################

mutable struct FixedEffectSolverCPU{T,M<:FixedEffectLinearMapCPU{T},C<:FixedEffectCoefficients{Matrix{T}}} <: AbstractFixedEffectSolver{T}
	m::M
	weights::AbstractVector
	b::Vector{T}
	r::Vector{T}
	x::C
	v::C
	h::C
	hbar::C
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:cpu}}) where {T}
	m = FixedEffectLinearMapCPU{T}(fes, weights)
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	blocks = m.plan.blocks
	x = FixedEffectCoefficients([zeros(T, block_width(block), block.nlevels) for block in blocks])
	v = FixedEffectCoefficients([zeros(T, block_width(block), block.nlevels) for block in blocks])
	h = FixedEffectCoefficients([zeros(T, block_width(block), block.nlevels) for block in blocks])
	hbar = FixedEffectCoefficients([zeros(T, block_width(block), block.nlevels) for block in blocks])
	return FixedEffectSolverCPU(m, weights, b, r, x, v, h, hbar)
end

function update_weights!(feM::FixedEffectSolverCPU{T}, weights::AbstractWeights) where {T}
	feM.m.plan = AbsorptionPlan(T, feM.m.plan, weights)
	feM.weights = weights
	return feM
end

function copy_internal!(feM::FixedEffectSolverCPU, field::Symbol, r::AbstractVector)
	copyto!(getfield(feM, field), r)
end

function copy_internal!(r::AbstractVector, feM::FixedEffectSolverCPU, field::Symbol)
	copyto!(r, getfield(feM, field))
end


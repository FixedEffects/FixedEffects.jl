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
	buffers::Vector{M}              # one k × n accumulator per thread
	ranges::Vector{UnitRange{Int}}  # contiguous row chunks
end
# For a block whose refs are sorted (see sorted_absorption_plan): row chunks
# end on group boundaries, so threads accumulate directly into the shared
# coefficient block — race-free without buffers, at any cardinality.
struct SortedGather
	ranges::Vector{UnitRange{Int}}  # row chunks aligned to group boundaries
end

mutable struct FixedEffectLinearMapCPU{T,F<:Vector{<:FixedEffect},P<:AbsorptionPlan,G<:AbstractVector} <: AbstractFixedEffectLinearMap{T}
	fes::F
	plan::P
	gathers::G
end

# The struct definitions must precede this constructor (a constructor method
# signature is evaluated at definition time, unlike ordinary function calls).
function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect}, plan::AbsorptionPlan,
		sorted_block::Integer) where {T}
	N = length(fes[1].refs)
	nt = nthreads()
	ranges = _row_chunks(N, nt)
	G = Union{SerialGather, SortedGather, ThreadedGather{Matrix{T}}}
	gathers = G[_gather_strategy(T, plan.blocks[j], N, nt, ranges, j == sorted_block)
		for j in eachindex(plan.blocks)]
	return FixedEffectLinearMapCPU{T,typeof(fes),typeof(plan),typeof(gathers)}(fes, plan, gathers)
end

function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect},
	weights::AbstractVector = uweights(T, length(fes[1].refs))) where {T}
	plan = AbsorptionPlan(T, fes, weights)
	return FixedEffectLinearMapCPU{T}(fes, plan, 0)
end

# Toggle to force the serial baseline (e.g. for benchmarking); threading is on by default.
const _USE_THREADED_GATHER = Ref(true)
# Threading the gather pays off only when the nt per-thread accumulators of size
# k × n fit in cache; beyond that the fill/merge memory traffic dominates
# and serial is faster.
const _GATHER_BUFFER_BUDGET = 8 * 1024 * 1024   # bytes
const _GATHER_MIN_ROWS = 100_000                # below this, threading overhead isn't worth it

# Per block: the designated sorted block gathers race-free over group-aligned
# chunks; otherwise thread only if the accumulators fit in cache and N is large.
function _gather_strategy(::Type{T}, block::AbsorbedBlock, N::Int, nt::Int,
		ranges::Vector{UnitRange{Int}}, sorted::Bool) where {T}
	k = block_width(block)
	if sorted && nt > 1 && N >= _GATHER_MIN_ROWS
		ranges_sorted = _group_aligned_chunks(block.refs, nt)
		# a dominant group can swallow most rows into one chunk and serialize
		# the pass; fall back to the buffered strategies in that case
		if maximum(length, ranges_sorted) <= 2 * cld(N, nt)
			return SortedGather(ranges_sorted)
		end
	end
	if _USE_THREADED_GATHER[] && nt > 1 && N >= _GATHER_MIN_ROWS &&
			nt * k * block.n * sizeof(T) <= _GATHER_BUFFER_BUDGET
		return ThreadedGather([zeros(T, k, block.n) for _ in 1:nt], ranges)
	else
		return SerialGather()
	end
end

# nchunks row chunks over sorted refs, each ending on a group boundary.
function _group_aligned_chunks(refs::AbstractVector, nchunks::Int)
	N = length(refs)
	ranges = Vector{UnitRange{Int}}(undef, nchunks)
	lo = 1
	for t in 1:nchunks
		if t == nchunks
			hi = N
		else
			target = div(N * t, nchunks)
			if target < lo
				hi = lo - 1          # empty chunk: the previous group swallowed its share
			else
				hi = searchsortedlast(refs, refs[target])
			end
		end
		ranges[t] = lo:hi
		lo = hi + 1
	end
	return ranges
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

# Sorted refs: each thread's chunk ends on a group boundary, so all threads
# write disjoint columns of coef_block — no buffers, no merge, any cardinality.
function gather_block!(coef_block::AbstractMatrix, block::AbsorbedBlock, qrows::AbstractMatrix,
	y::AbstractVector, α::Number, g::SortedGather)
	@threads for t in eachindex(g.ranges)
		_gather_block!(coef_block, block, qrows, y, α, g.ranges[t])
	end
	return coef_block
end

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
## 1d) Fused bidiagonalization step (single-pass LSMR iterations)
##
## u ← A v + c u, β = ‖u‖, g ← A'u — in as few passes over u as the gather
## strategies allow. The mul!-based lsmr! iteration streams u about seven
## times (one scatter per block, each reading and writing all of u, a norm
## pass, a scale pass, one gather read per block); here the scatters of every
## block, the norm, and — when every block has cache-resident per-thread
## buffers — the gathers all run inside one loop over observations, so u and
## refs/qrows are streamed once per iteration.
##
## Blocks are passed as tuples: the recursive helpers unroll across blocks and
## the per-column loops stay compile-time (block_width, as in the kernels
## above). Building the tuples is dynamic, but it happens once per call and
## the kernels behind the function barrier specialize.
##
##############################################################################

function bidiag_forward!(u::Vector{T}, g::FixedEffectCoefficients, fem::FixedEffectLinearMapCPU{T},
		v::FixedEffectCoefficients, c::Number) where {T}
	fill!(g, zero(T))
	blocks = Tuple(fem.plan.blocks)
	qrowss = Tuple(fem.plan.qrows)
	vs = Tuple(v.x)
	gs = Tuple(g.x)
	N = length(u)
	nt = nthreads()
	if nt > 1 && N >= _GATHER_MIN_ROWS && all(gather -> gather isa Union{SortedGather, ThreadedGather}, fem.gathers)
		# fully fused: each thread owns a row chunk and accumulates every
		# block's gather — directly into g for the sorted block (its chunks
		# end on group boundaries, so writes are disjoint), into private
		# buffers for the others — plus a norm partial. When a sorted block
		# is present its group-aligned chunks are used for the whole pass.
		gathers = Tuple(fem.gathers)
		sorted = findfirst(gather -> gather isa SortedGather, fem.gathers)
		if sorted === nothing
			ranges = _row_chunks(N, nt)
		else
			ranges = fem.gathers[sorted].ranges
		end
		partials = Vector{Float64}(undef, length(ranges))
		@threads for t in eachindex(ranges)
			targets = map(gathers, gs) do gather, coef_block
				if gather isa SortedGather
					coef_block
				else
					buf = gather.buffers[t]
					fill!(buf, zero(T))
					buf
				end
			end
			partials[t] = _bidiag_chunk!(u, targets, blocks, qrowss, vs, T(c), ranges[t])
		end
		@inbounds for (coef_block, gather) in zip(g.x, fem.gathers)
			gather isa ThreadedGather || continue
			for buf in gather.buffers
				@simd for idx in eachindex(coef_block)
					coef_block[idx] += buf[idx]
				end
			end
		end
		s = sum(partials)
	elseif nt > 1 && N >= _GATHER_MIN_ROWS
		# fused scatters + norm in one threaded pass; the gathers of each
		# block then run through their existing strategies on the raw u
		ranges = _row_chunks(N, nt)
		partials = Vector{Float64}(undef, length(ranges))
		@threads for t in eachindex(ranges)
			partials[t] = _bidiag_scatter_chunk!(u, blocks, qrowss, vs, T(c), ranges[t])
		end
		s = sum(partials)
		for (coef_block, block, qrows, gather) in zip(g.x, fem.plan.blocks, fem.plan.qrows, fem.gathers)
			gather_block!(coef_block, block, qrows, u, one(T), gather)
		end
	else
		# serial: everything in one pass, gathers accumulated directly into g
		s = _bidiag_chunk!(u, gs, blocks, qrowss, vs, T(c), eachindex(u))
	end
	return T(sqrt(s))
end

# u[i] ← c * u[i] + Σ_blocks fit_i; accumulates each block's gather into
# `bufs` (g's own blocks when serial, one thread's private buffers when
# threaded) and returns the Float64 sum of squares of the updated u.
function _bidiag_chunk!(u::Vector{T}, bufs::Tuple, blocks::Tuple, qrowss::Tuple, vs::Tuple,
		c::T, range) where {T}
	s = 0.0
	@inbounds for i in range
		ui = c * u[i] + _scatter_fit(blocks, qrowss, vs, i)
		u[i] = ui
		s += abs2(Float64(ui))
		_gather_accum!(bufs, blocks, qrowss, i, ui)
	end
	return s
end

function _bidiag_scatter_chunk!(u::Vector{T}, blocks::Tuple, qrowss::Tuple, vs::Tuple,
		c::T, range) where {T}
	s = 0.0
	@inbounds for i in range
		ui = c * u[i] + _scatter_fit(blocks, qrowss, vs, i)
		u[i] = ui
		s += abs2(Float64(ui))
	end
	return s
end

# Recursion over the block tuples: each level specializes on its block type,
# so the inner loops over columns unroll (block_width is compile-time).
@inline _scatter_fit(::Tuple{}, ::Tuple{}, ::Tuple{}, i) = false   # additive zero of any float type
@inline function _scatter_fit(blocks::Tuple, qrowss::Tuple, vs::Tuple, i)
	block = first(blocks)
	qrows = first(qrowss)
	vcoef = first(vs)
	@inbounds begin
		gr = block.refs[i]
		fit = zero(eltype(qrows))
		for col in 1:block_width(block)
			fit += vcoef[col, gr] * qrows[col, i]
		end
	end
	return fit + _scatter_fit(Base.tail(blocks), Base.tail(qrowss), Base.tail(vs), i)
end

@inline _gather_accum!(::Tuple{}, ::Tuple{}, ::Tuple{}, i, ui) = nothing
@inline function _gather_accum!(bufs::Tuple, blocks::Tuple, qrowss::Tuple, i, ui)
	buf = first(bufs)
	block = first(blocks)
	qrows = first(qrowss)
	@inbounds begin
		gr = block.refs[i]
		for col in 1:block_width(block)
			buf[col, gr] += ui * qrows[col, i]
		end
	end
	return _gather_accum!(Base.tail(bufs), Base.tail(blocks), Base.tail(qrowss), i, ui)
end

##############################################################################
##
## 2. FixedEffectSolverCPU
##
##############################################################################

mutable struct FixedEffectSolverCPU{T,M<:FixedEffectLinearMapCPU{T},C<:FixedEffectCoefficients{Matrix{T}}} <: AbstractFixedEffectSolver{T}
	m::M
	weights::AbstractVector
	perm::Union{Nothing, Vector{Int}}   # observation order of the internal storage (see sorted_absorption_plan)
	b::Vector{T}
	r::Vector{T}
	x::C
	v::C
	h::C
	hbar::C
	g::C
end

function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:cpu}}) where {T}
	plan, perm, sorted_block = sorted_absorption_plan(T, fes, weights)
	m = FixedEffectLinearMapCPU{T}(fes, plan, sorted_block)
	if perm !== nothing
		weights = _permute_weights(weights, perm)
	end
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	blocks = m.plan.blocks
	x = FixedEffectCoefficients([zeros(T, block_width(block), block.n) for block in blocks])
	v = FixedEffectCoefficients([zeros(T, block_width(block), block.n) for block in blocks])
	h = FixedEffectCoefficients([zeros(T, block_width(block), block.n) for block in blocks])
	hbar = FixedEffectCoefficients([zeros(T, block_width(block), block.n) for block in blocks])
	g = FixedEffectCoefficients([zeros(T, block_width(block), block.n) for block in blocks])
	return FixedEffectSolverCPU(m, weights, perm, b, r, x, v, h, hbar, g)
end

function update_weights!(feM::FixedEffectSolverCPU{T}, weights::AbstractWeights) where {T}
	if feM.perm !== nothing
		weights = _permute_weights(weights, feM.perm)
	end
	feM.m.plan = AbsorptionPlan(T, feM.m.plan, weights)
	feM.weights = weights
	return feM
end

# Internal storage lives in the (possibly) sorted observation order; these
# translate from and to the caller's order. The permutation loops sit behind
# function barriers: getfield with a runtime Symbol is abstractly typed, and
# an element-wise loop on it would dispatch dynamically on every element.
function copy_internal!(feM::FixedEffectSolverCPU, field::Symbol, r::AbstractVector)
	dest = getfield(feM, field)
	if feM.perm === nothing
		copyto!(dest, r)
	else
		_gather_perm!(dest, r, feM.perm)
	end
	return dest
end

function copy_internal!(r::AbstractVector, feM::FixedEffectSolverCPU, field::Symbol)
	src = getfield(feM, field)
	if feM.perm === nothing
		copyto!(r, src)
	else
		_scatter_perm!(r, src, feM.perm)
	end
	return r
end

# dest[i] = r[perm[i]]
function _gather_perm!(dest::AbstractVector, r::AbstractVector, perm::Vector{Int})
	@inbounds for i in eachindex(dest)
		dest[i] = r[perm[i]]
	end
	return dest
end

# r[perm[i]] = src[i]
function _scatter_perm!(r::AbstractVector, src::AbstractVector, perm::Vector{Int})
	@inbounds for i in eachindex(src)
		r[perm[i]] = src[i]
	end
	return r
end

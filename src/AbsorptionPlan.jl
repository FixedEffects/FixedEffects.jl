##############################################################################
##
## AbsorptionPlan
##
##############################################################################

## 1a) Types

# One physical operator unit: all fixed-effect terms that share the same refs,
# e.g. fe(id) + fe(id)&x is one block of width 2 with interactions (1, x).
struct AbsorbedBlock{R<:AbstractVector{<:Integer},I<:Tuple}
	refs::R
	interactions::I           # one column per term: 1, x, x^2, ...
	n::Int                    # number of groups (= fe.n of the input terms)
	input_terms::Vector{Int}  # indices into the original fes vector, one per column
end

"""
Whitened representation of a set of fixed-effect terms, grouped into
`AbsorbedBlock`s of terms sharing the same refs.

For block `j` with columns `Z` (the interactions) and group `g`, let
`G_g = Z_g' W Z_g` be the weighted Gram matrix. The plan stores a rank-revealing
transform `R_g = transforms[j][:, 1:ranks[j][g], g]` with `R_g' G_g R_g = I`,
and the whitened rows `qrows[j][c, i] = sqrt(w_i) z_i' R_g[:, c]`, whose
columns are orthonormal within each group.

The solvers apply the whitened operator `A = W^(1/2) Z R`: column scaling
(`R` and `sqrt(w)`) is baked into `qrows` here, while `solve_residuals!` and
`solve_coefficients!` scale the RHS by `sqrt(w)` at solve time — both scalings
are needed, and neither may be applied twice. Whitened coefficients transform
back to original coordinates via `β_g = R_g θ_g` (see `recover_coefficients`).
For a single scalar fixed effect this reduces to the usual Jacobi (diagonal)
preconditioner.
"""
struct AbsorptionPlan{B<:AbstractVector{<:AbsorbedBlock},TR<:AbstractVector,RA<:AbstractVector,RV<:AbstractVector}
	blocks::B
	transforms::TR            # per block: k × k × n, R_g in columns 1:ranks[g]
	ranks::RA                 # per block: rank of each group's Gram matrix
	qrows::RV                 # per block: k × nobs whitened rows sqrt(w) Z R
end

block_width(block::AbsorbedBlock) = length(block.interactions)

## 1b) Constructors

function AbsorptionPlan(::Type{T}, fes::Vector{<:FixedEffect}, weights::AbstractVector;
		ranktol::Union{Nothing,Real} = nothing) where {T}
	blocks = _build_absorbed_blocks(fes)
	transforms, ranks, qrows = _build_transforms(T, blocks, weights, ranktol)
	return AbsorptionPlan(blocks, transforms, ranks, qrows)
end

# Rebuild for new weights, reusing the block structure (which does not depend on weights).
function AbsorptionPlan(::Type{T}, plan::AbsorptionPlan, weights::AbstractVector;
		ranktol::Union{Nothing,Real} = nothing) where {T}
	transforms, ranks, qrows = _build_transforms(T, plan.blocks, weights, ranktol)
	return AbsorptionPlan(plan.blocks, transforms, ranks, qrows)
end

## 1b') Sorted observation layout

# Sorting observations by one block's groups turns that block's random
# coefficient accesses into streams in every scatter/gather, and lets
# its gather run race-free without per-thread buffers (row chunks can end on
# group boundaries). Sorting pays once the block's coefficient tile outgrows
# the caches; below this threshold it is skipped. Set to 0 to force sorting
# (used by the tests).
const _SORT_TILE_BYTES = Ref(1024 * 1024)

# Build the plan on a sorted observation order: pick the block with the
# largest coefficient tile (the largest random-access working set) and, when
# its refs are unsorted and that tile is large enough to matter, permute every
# block's refs and interactions, and the weights, by its counting-sort order.
# Returns (plan, perm, sorted_block):
# perm is nothing when the observations were not permuted (sorted_block may
# still name an already-sorted block; 0 when none is sorted). Callers must
# permute solver-side observation data (weights, right-hand sides) with perm.
function sorted_absorption_plan(::Type{T}, fes::Vector{<:FixedEffect}, weights::AbstractVector;
		ranktol::Union{Nothing,Real} = nothing) where {T}
	blocks = _build_absorbed_blocks(fes)
	perm = nothing
	sorted_block = 0
	# with a single block the solvers use one direct projection: no iterations
	# to speed up, so sorting would only add the permutation passes
	if length(blocks) > 1
		j = argmax([block_width(block) * block.n for block in blocks])
		if block_width(blocks[j]) * blocks[j].n * sizeof(T) > _SORT_TILE_BYTES[]
			if issorted(blocks[j].refs)
				sorted_block = j
			else
				_, _, perm = _group_permutation(blocks[j].refs, blocks[j].n)
				blocks = [_permute_block(block, perm) for block in blocks]
				weights = _permute_weights(weights, perm)
				sorted_block = j
			end
		end
	end
	transforms, ranks, qrows = _build_transforms(T, blocks, weights, ranktol)
	return AbsorptionPlan(blocks, transforms, ranks, qrows), perm, sorted_block
end

function _permute_block(block::AbsorbedBlock, perm::Vector{Int})
	interactions = map(block.interactions) do interaction
		if interaction isa UnitWeights
			interaction
		else
			interaction[perm]
		end
	end
	return AbsorbedBlock(block.refs[perm], interactions, block.n, block.input_terms)
end

_permute_weights(weights::UnitWeights, ::Vector{Int}) = weights
_permute_weights(weights::AbstractVector, perm::Vector{Int}) = weights[perm]

function _build_absorbed_blocks(fes::Vector{<:FixedEffect})
	blocks = AbsorbedBlock[]
	for (j, fe) in enumerate(fes)
		block_id = findfirst(block -> block.n == fe.n && block.refs == fe.refs, blocks)
		if block_id === nothing
			push!(blocks, AbsorbedBlock(fe.refs, (fe.interaction,), fe.n, [j]))
		else
			block = blocks[block_id]
			interactions = (block.interactions..., fe.interaction)
			input_terms = copy(block.input_terms)
			push!(input_terms, j)
			blocks[block_id] = AbsorbedBlock(block.refs, interactions, block.n, input_terms)
		end
	end
	return blocks
end

function _build_transforms(::Type{T}, blocks::AbstractVector{<:AbsorbedBlock},
		weights::AbstractVector, ranktol::Union{Nothing,Real}) where {T}
	transforms = Vector{Array{T,3}}(undef, length(blocks))
	ranks = Vector{Vector{Int}}(undef, length(blocks))
	qrows = Vector{Matrix{T}}(undef, length(blocks))
	for j in eachindex(blocks)
		transforms[j], ranks[j], qrows[j] = _build_block_transform(T, blocks[j], weights, ranktol)
	end
	return transforms, ranks, qrows
end

## 1c) Per-block transform build (rank-revealing Gram-Schmidt)

function _build_block_transform(::Type{T}, block::AbsorbedBlock, weights::AbstractVector,
		ranktol::Union{Nothing,Real}) where {T}
	k = block_width(block)
	nlevels = block.n
	nobs = length(block.refs)
	transforms = zeros(T, k, k, nlevels)
	ranks = zeros(Int, nlevels)
	qrows = zeros(T, k, nobs)
	if k == 1
		interaction = block.interactions[1]
		gram = zeros(T, nlevels)
		@inbounds for i in eachindex(block.refs)
			g = block.refs[i]
			z = T(interaction[i])
			gram[g] += T(weights[i]) * abs2(z)
		end
		@inbounds for g in 1:nlevels
			if gram[g] > zero(T)
				transforms[1, 1, g] = inv(sqrt(gram[g]))
				ranks[g] = 1
			end
		end
		@spawn_for_chunks 100_000 for i in eachindex(block.refs)
			@inbounds begin
				g = block.refs[i]
				qrows[1, i] = sqrt(T(weights[i])) * T(interaction[i]) * transforms[1, 1, g]
			end
		end
		return transforms, ranks, qrows
	elseif k == 2 && count(interaction -> interaction isa UnitWeights, block.interactions) == 1
		# Stable weighted moments avoid cancellation in within-group slope variation.
		intercept = block.interactions[1] isa UnitWeights ? 1 : 2
		slope = 3 - intercept
		z = block.interactions[slope]
		sumw = zeros(T, nlevels)
		anchor = zeros(T, nlevels)
		mean = zeros(T, nlevels)
		m2 = zeros(T, nlevels)
		@inbounds for i in eachindex(block.refs)
			g = block.refs[i]
			w = T(weights[i])
			zi = T(z[i])
			new_sumw = sumw[g] + w
			if new_sumw > zero(T)
				if iszero(sumw[g])
					anchor[g] = zi
				end
				centered_z = zi - anchor[g]
				delta = centered_z - mean[g]
				new_mean = mean[g] + w * delta / new_sumw
				m2[g] += w * delta * (centered_z - new_mean)
				mean[g] = new_mean
				sumw[g] = new_sumw
			end
		end

		tol = ranktol === nothing ? T(2) * sqrt(eps(T)) : T(ranktol)
		@inbounds for g in 1:nlevels
			if sumw[g] > zero(T) && one(T) > tol
				transforms[intercept, 1, g] = inv(sqrt(sumw[g]))
				ranks[g] = 1
			end
			centered_sumsq = max(zero(T), m2[g])
			group_mean = anchor[g] + mean[g]
			slope_sumsq = centered_sumsq + sumw[g] * abs2(group_mean)
			if ranks[g] == 1 && slope_sumsq > zero(T) &&
					sqrt(centered_sumsq / slope_sumsq) > tol
				transforms[slope, 2, g] = inv(sqrt(centered_sumsq))
				transforms[intercept, 2, g] = -group_mean * transforms[slope, 2, g]
				ranks[g] = 2
			end
		end
		@inbounds for i in eachindex(block.refs)
			g = block.refs[i]
			sqrtw = sqrt(T(weights[i]))
			qrows[1, i] = sqrtw * transforms[intercept, 1, g]
			qrows[2, i] = sqrtw * (T(z[i]) - anchor[g] - mean[g]) * transforms[slope, 2, g]
		end
		return transforms, ranks, qrows
	else
		counts, offsets, perm = _group_permutation(block.refs, nlevels)
		maxrows = isempty(counts) ? 0 : maximum(counts)
		if nthreads() > 1 && nobs >= 100_000
			nchunks = max(1, min(nthreads(), nlevels))
		else
			nchunks = 1
		end
		if nchunks == 1
			_build_block_transform_chunk!(transforms, ranks, qrows, block, weights,
				ranktol, counts, offsets, perm, 1:nlevels, maxrows)
		else
			# Groups are disjoint row segments of perm, so chunks can be processed in parallel.
			@sync for chunk in _row_chunks(nlevels, nchunks)
				let chunk = chunk
					Base.Threads.@spawn _build_block_transform_chunk!(transforms, ranks, qrows,
						block, weights, ranktol, counts, offsets, perm, chunk, maxrows)
				end
			end
		end
		return transforms, ranks, qrows
	end
end

function _build_block_transform_chunk!(transforms::AbstractArray{T,3}, ranks::AbstractVector{Int},
		qrows::AbstractMatrix{T}, block::AbsorbedBlock, weights::AbstractVector,
		ranktol::Union{Nothing,Real}, counts::AbstractVector{Int}, offsets::AbstractVector{Int},
		perm::AbstractVector{Int}, chunk, maxrows::Int) where {T}
	k = block_width(block)
	tmp = Vector{T}(undef, maxrows)
	colnorms = Vector{T}(undef, k)
	coef = zeros(T, k)
	for g in chunk
		firstrow = offsets[g]
		lastrow = offsets[g + 1] - 1
		nrows = counts[g]
		nrows > 0 || continue
		tol = ranktol === nothing ? T(k) * sqrt(eps(T)) : T(ranktol)
		@inbounds for a in 1:k
			s = zero(T)
			zvals = block.interactions[a]
			for p in firstrow:lastrow
				i = perm[p]
				z = T(zvals[i])
				s += T(weights[i]) * abs2(z)
			end
			colnorms[a] = sqrt(s)
		end
		rank = 0
		for a in 1:k
			colnorm = colnorms[a]
			colnorm > zero(T) || continue
			invcolnorm = inv(colnorm)
			fill!(coef, zero(T))
			coef[a] = invcolnorm
			zvals = block.interactions[a]
			@inbounds for (localrow, p) in enumerate(firstrow:lastrow)
				i = perm[p]
				tmp[localrow] = sqrt(T(weights[i])) * T(zvals[i]) * invcolnorm
			end
			for _ in 1:2
				for c in 1:rank
					h = zero(T)
					@inbounds for (localrow, p) in enumerate(firstrow:lastrow)
						i = perm[p]
						h += qrows[c, i] * tmp[localrow]
					end
					iszero(h) && continue
					@inbounds for (localrow, p) in enumerate(firstrow:lastrow)
						i = perm[p]
						tmp[localrow] -= h * qrows[c, i]
					end
					@inbounds for b in 1:k
						coef[b] -= h * transforms[b, c, g]
					end
				end
			end
			nrm2 = zero(T)
			@inbounds for localrow in 1:nrows
				nrm2 += abs2(tmp[localrow])
			end
			nrm = sqrt(nrm2)
			if nrm > tol
				rank += 1
				invnrm = inv(nrm)
				@inbounds for (localrow, p) in enumerate(firstrow:lastrow)
					i = perm[p]
					qrows[rank, i] = tmp[localrow] * invnrm
				end
				@inbounds for b in 1:k
					transforms[b, rank, g] = coef[b] * invnrm
				end
			end
		end
		ranks[g] = rank
	end
	return nothing
end

function _group_permutation(refs::AbstractVector, nlevels::Integer)
	counts = zeros(Int, nlevels)
	@inbounds for i in eachindex(refs)
		counts[refs[i]] += 1
	end
	offsets = Vector{Int}(undef, nlevels + 1)
	offsets[1] = 1
	@inbounds for g in 1:nlevels
		offsets[g + 1] = offsets[g] + counts[g]
	end
	cursor = copy(offsets)
	perm = Vector{Int}(undef, length(refs))
	@inbounds for i in eachindex(refs)
		g = refs[i]
		p = cursor[g]
		perm[p] = i
		cursor[g] = p + 1
	end
	return counts, offsets, perm
end

function _row_chunks(n::Int, k::Int)
	base, rem = divrem(n, k)
	out = Vector{UnitRange{Int}}(undef, k)
	s = 1
	for t in 1:k
		len = base + (t <= rem ? 1 : 0)
		out[t] = s:(s + len - 1)
		s += len
	end
	return out
end

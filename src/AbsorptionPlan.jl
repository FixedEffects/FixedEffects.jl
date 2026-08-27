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
_ncoef(plan::AbsorptionPlan) = sum(block_width(block) * block.n for block in plan.blocks)

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
	end

	counts, offsets, perm = _group_permutation(block.refs, nlevels)
	maxrows = isempty(counts) ? 0 : maximum(counts)
	if nthreads() > 1 && nobs >= 100_000
		nchunks = max(1, min(nthreads(), nlevels))
	else
		nchunks = 1
	end
	chunks = _row_chunks(nlevels, nchunks)
	if nchunks == 1
		_build_block_transform_chunk!(transforms, ranks, qrows, block, weights,
			ranktol, counts, offsets, perm, chunks[1], maxrows)
	else
		# Groups are disjoint row segments of perm, so chunks can be processed in parallel.
		@sync for chunk in chunks
			let chunk = chunk
				Base.Threads.@spawn _build_block_transform_chunk!(transforms, ranks, qrows,
					block, weights, ranktol, counts, offsets, perm, chunk, maxrows)
			end
		end
	end
	return transforms, ranks, qrows
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


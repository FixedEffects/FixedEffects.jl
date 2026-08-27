##############################################################################
##
## Implement AbstractFixedEffectLinearMap
##
##############################################################################

mutable struct FixedEffectLinearMapCPU{T,F<:Vector{<:FixedEffect},S<:AbstractVector,C<:AbstractVector,G<:AbstractVector} <: AbstractFixedEffectLinearMap{T}
	fes::F
	scales::S
	caches::C
	gathers::G
end

# Toggle to force the serial baseline (e.g. for benchmarking); threading is on by default.
const _USE_THREADED_GATHER = Ref(true)
# Threading the gather pays off only when the nt per-thread accumulators of length fe.n fit
# in cache; beyond that the fill/merge memory traffic dominates and serial is faster.
const _GATHER_BUFFER_BUDGET = 8 * 1024 * 1024   # bytes
const _GATHER_MIN_ROWS = 100_000                # below this, threading overhead isn't worth it

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

# Per fixed effect, thread the gather only if the accumulators fit in cache and N is large.
function _gather_strategy(::Type{T}, fe::FixedEffect, N::Int, nt::Int,
		ranges::Vector{UnitRange{Int}}) where {T}
	if _USE_THREADED_GATHER[] && nt > 1 && N >= _GATHER_MIN_ROWS &&
			nt * fe.n * sizeof(T) <= _GATHER_BUFFER_BUDGET
		return ThreadedGather([zeros(T, fe.n) for _ in 1:nt], ranges)
	else
		return SerialGather()
	end
end

function FixedEffectLinearMapCPU{T}(fes::Vector{<:FixedEffect}) where {T}
	scales = [zeros(T, fe.n) for fe in fes]
	caches = [zeros(T, length(fes[1].interaction)) for fe in fes]
	N = length(fes[1].refs)
	nt = nthreads()
	ranges = _row_chunks(N, nt)
	G = Union{SerialGather, ThreadedGather{Vector{T}}}
	gathers = G[_gather_strategy(T, fe, N, nt, ranges) for fe in fes]
	return FixedEffectLinearMapCPU{T,typeof(fes),typeof(scales),typeof(caches),typeof(gathers)}(fes, scales, caches, gathers)
end


# The one place the gather arithmetic lives: scatter-add a row range into `out`,
# out[refs[i]] += α * y[i] * cache[i]. No @simd: distinct i may write the same out[refs[i]].
function _gather!(out::AbstractVector, refs::AbstractVector, α::Number,
	y::AbstractVector, cache::AbstractVector, range)
	@fastmath @inbounds for i in range
		out[refs[i]] += α * y[i] * cache[i]
	end
	return out
end

# Serial: one pass over all rows straight into fecoef (which already holds β * old).
gather!(fecoef::AbstractVector, refs::AbstractVector, α::Number,
	y::AbstractVector, cache::AbstractVector, ::SerialGather) =
	_gather!(fecoef, refs, α, y, cache, eachindex(y))

# Threaded: each thread reduces its row chunk into a private (cache-resident) buffer,
# then the buffers are summed into fecoef.
function gather!(fecoef::AbstractVector, refs::AbstractVector, α::Number,
	y::AbstractVector, cache::AbstractVector, g::ThreadedGather)
	@threads for t in eachindex(g.buffers)
		buf = g.buffers[t]
		fill!(buf, zero(eltype(buf)))
		_gather!(buf, refs, α, y, cache, g.ranges[t])
	end
	@inbounds for buf in g.buffers
		@simd for k in eachindex(fecoef)
			fecoef[k] += buf[k]
		end
	end
	return fecoef
end

function scatter!(y::AbstractVector, α::Number, fecoef::AbstractVector,
	refs::AbstractVector, cache::AbstractVector)
	@spawn_for_chunks 100_000 for i in eachindex(y)
		@inbounds y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

function scatter!(y::Vector, α::Number, fecoef::Vector,
	refs::Vector, cache::Vector, β::Number)
	if iszero(β)
		@spawn_for_chunks 100_000 for i in eachindex(y)
			@inbounds y[i] = α * fecoef[refs[i]] * cache[i]
		end
	elseif isone(β)
		scatter!(y, α, fecoef, refs, cache)
	else
		@spawn_for_chunks 100_000 for i in eachindex(y)
			# Fuse y[i] = β * y[i] + α * fecoef[refs[i]] * cache[i] to avoid a separate scaling pass.
			@inbounds y[i] = β * y[i] + α * fecoef[refs[i]] * cache[i]
		end
	end
end


##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

mutable struct FixedEffectSolverCPU{T,M<:FixedEffectLinearMapCPU{T},C<:FixedEffectCoefficients{Vector{T}}} <: AbstractFixedEffectSolver{T}
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
	m = FixedEffectLinearMapCPU{T}(fes)
	b = zeros(T, length(weights))
	r = zeros(T, length(weights))
	x = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([zeros(T, fe.n) for fe in fes])
	return update_weights!(FixedEffectSolverCPU(m, weights, b, r, x, v, h, hbar), weights)
end


function update_weights!(feM::FixedEffectSolverCPU, weights::AbstractWeights)
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, weights)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, weights, scale)
	end
	feM.weights = weights
	return feM
end

function scale!(scale::AbstractVector, refs::AbstractVector, interaction::AbstractVector, weights::AbstractVector)
    fill!(scale, 0)
	# no @simd: multiple i may write to the same scale[refs[i]]
	@fastmath @inbounds for i in eachindex(refs)
		scale[refs[i]] += abs2(interaction[i]) * weights[i]
	end
	# Case of interaction variable equal to zero in the category (issue #97)
	T = eltype(scale)
	@fastmath @inbounds @simd for i in eachindex(scale)
	    scale[i] = scale[i] > 0 ? (1 / sqrt(scale[i])) : zero(T)
	end
end

function cache!(cache::AbstractVector, refs::AbstractVector, interaction::AbstractVector, weights::AbstractVector, scale::AbstractVector)
	@spawn_for_chunks 100_000 for i in eachindex(cache)
		@inbounds @fastmath cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
end

function copy_internal!(feM::FixedEffectSolverCPU, field::Symbol, r::AbstractVector)
	copyto!(getfield(feM, field), r)
end

function copy_internal!(r::AbstractVector, feM::FixedEffectSolverCPU, field::Symbol)
	copyto!(r, getfield(feM, field))
end

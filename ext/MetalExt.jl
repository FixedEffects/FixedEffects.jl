module MetalExt
using FixedEffects, Metal
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!, lsmr!, AbstractFixedEffectLinearMap
Metal.allowscalar(false)

##############################################################################
##
## Conversion FixedEffect between CPU and Metal
##
##############################################################################

function _mtl(T::Type, fe::FixedEffect)
	refs = MtlArray(fe.refs)
	interaction = _mtl(T, fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
_mtl(T::Type, w::UnitWeights) = Metal.ones(T, length(w))
_mtl(T::Type, w::AbstractVector) = MtlVector{T}(convert(Vector{T}, w))

##############################################################################
##
## FixedEffectLinearMap on Metal
##
## Model matrix of categorical variables
## mutiplied by diag(1/sqrt(∑w * interaction^2, ..., ∑w * interaction^2) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## eltype
## size
## mul!
##
##############################################################################

mutable struct FixedEffectLinearMapMetal{T} <: AbstractFixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	scales::Vector{<:AbstractVector}
	caches::Vector
	nthreads::Int
end

function bucketize_refs(refs::Vector, n::Int)
	# count the number of obs per group
    counts = zeros(UInt32, n)
    @inbounds for r in refs
        counts[r] += 0x00000001
    end
	# offsets is vcat(1, cumsum(counts))
    offsets = Vector{UInt32}(undef, n + 1)
    offsets[1] = 0x00000001
    @inbounds for k in 1:n
        offsets[k+1] = offsets[k] + counts[k]
    end
    next = offsets[1:n]
    perm = Vector{UInt32}(undef, length(refs))
    @inbounds for i in eachindex(refs)
        r = refs[i]
        p = next[r]
        perm[p] = UInt32(i)
        next[r] = p + 0x00000001
    end
    return perm, offsets
end

function FixedEffectLinearMapMetal{T}(fes::Vector{<:FixedEffect}, nthreads) where {T}
	fes2 = [_mtl(T, fe) for fe in fes]
	scales = [Metal.zeros(T, fe.n) for fe in fes]
	caches = [[Metal.zeros(T, length(fe.refs)), Metal.zeros(UInt32, 1), Metal.zeros(UInt32, 1)] for fe in fes]
	Threads.@threads for i in 1:length(fes)
		refs = fes[i].refs
		n = fes[i].n
		if n < min(100_000,  div(length(refs), 16))	
			out = bucketize_refs(refs, n)
			caches[i][2] = MtlArray(out[1])
			caches[i][3] = MtlArray(out[2])
		end
	end
	return FixedEffectLinearMapMetal{T}(fes2, scales, caches, nthreads)
end

function FixedEffects.gather!(fecoef::MtlVector, refs::MtlVector, α::Number, y::MtlVector, cache::Vector, nthreads::Integer)
	n = length(fecoef)
	if n < min(100_000,  div(length(refs), 16))
		Metal.@sync @metal threads=nthreads groups=n gather_kernel_bin!(fecoef, refs, α, y, cache[1], cache[2], cache[3], Val(nthreads))
	else
		nblocks = cld(length(y), nthreads)
		Metal.@sync @metal threads=nthreads groups=nblocks gather_kernel!(fecoef, refs, α, y, cache[1])
	end
end

function gather_kernel_bin!(fecoef, refs, α, y, cache, perm, offsets, ::Val{NT}) where {NT}
    k   = threadgroup_position_in_grid().x          # 1..K (Julia-style indexing) :contentReference[oaicite:2]{index=2}
    tid = thread_position_in_threadgroup().x        # 1..nthreads :contentReference[oaicite:3]{index=3}
    nt  = threads_per_threadgroup().x               # nthreads :contentReference[oaicite:4]{index=4}

    # threadgroup scratch
    T = eltype(fecoef)
    shared = Metal.MtlThreadGroupArray(T, NT)  # threadgroup-local array :contentReference[oaicite:5]{index=5}

    start = @inbounds offsets[k]
    stop  = @inbounds offsets[k+1] - Int32(1)

    acc = zero(T)

    # each thread walks its portion of the bucket
    j = start + UInt32(tid - 1)
    while j <= stop
        i = @inbounds perm[j]
        @inbounds acc += (α * y[i] * cache[i])
        j += UInt32(nt)
    end

    @inbounds shared[tid] = acc
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)  # sync + tg fence :contentReference[oaicite:6]{index=6}

    # tree reduction in shared memory
    offset = UInt32(nt ÷ 0x00000002)
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid + offset]
        end
        Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        offset ÷= 0x00000002
    end

    # one write per coefficient (no atomics needed if groups == n and 1 group per k)
    if tid == 0x00000001
        @inbounds fecoef[k] += shared[1]
    end

    return nothing
end

function gather_kernel!(fecoef, refs, α, y, cache)
	i = thread_position_in_grid_1d()
	if i <= length(refs)
		@inbounds Metal.atomic_fetch_add_explicit(pointer(fecoef, refs[i]), α * y[i] * cache[i])
	end
	return nothing
end

function FixedEffects.scatter!(y::MtlVector, α::Number, fecoef::MtlVector, refs::MtlVector, cache::Vector, nthreads::Integer)
	nblocks = cld(length(y), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks scatter_kernel!(y, α, fecoef, refs, cache[1])
end

function scatter_kernel!(y, α, fecoef, refs, cache)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		@inbounds y[i] += α * fecoef[refs[i]] * cache[i]
	end
	return nothing
end



##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

mutable struct FixedEffectSolverMetal{T} <: FixedEffects.AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapMetal{T}
	weights::MtlVector{T}
	b::MtlVector{T}
	r::MtlVector{T}
	x::FixedEffectCoefficients{<: AbstractVector{T}}
	v::FixedEffectCoefficients{<: AbstractVector{T}}
	h::FixedEffectCoefficients{<: AbstractVector{T}}
	hbar::FixedEffectCoefficients{<: AbstractVector{T}}
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
	fes::Vector{<:FixedEffect}
end
	
function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:Metal}}, nthreads = nothing) where {T}
	if nthreads === nothing
		nthreads = Int(device().maxThreadsPerThreadgroup.width)
	end
	nthreads = prevpow(2, nthreads)
	m = FixedEffectLinearMapMetal{T}(fes, nthreads)
	b = Metal.zeros(T, length(weights))
	r = Metal.zeros(T, length(weights))
	x = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([Metal.zeros(T, fe.n) for fe in fes])
	tmp = zeros(T, length(weights))
	feM = FixedEffectSolverMetal{T}(m, Metal.zeros(T, length(weights)), b, r, x, v, h, hbar, tmp, fes)
	FixedEffects.update_weights!(feM, weights)
end


function FixedEffects.update_weights!(feM::FixedEffectSolverMetal{T}, weights::AbstractWeights) where {T}
	copyto!(feM.weights, _mtl(T, weights))
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, feM.weights, feM.m.nthreads)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, feM.weights, scale, feM.m.nthreads)
	end	
	return feM
end

function scale!(scale::MtlVector, refs::MtlVector, interaction::MtlVector, weights::MtlVector, nthreads::Integer)
	nblocks = cld(length(refs), nthreads) 
    fill!(scale, 0)
	Metal.@sync @metal threads=nthreads groups=nblocks scale_kernel!(scale, refs, interaction, weights)
	Metal.@sync @metal threads=nthreads groups=nblocks inv_kernel!(scale, eltype(scale))
end

function scale_kernel!(scale, refs, interaction, weights)
	i = thread_position_in_grid_1d()
	if i <= length(refs)
		@inbounds Metal.atomic_fetch_add_explicit(pointer(scale, refs[i]), interaction[i]^2 * weights[i])
	end
	return nothing
end

function inv_kernel!(scale, T)
	i = thread_position_in_grid_1d()
	if i <= length(scale)
		@inbounds scale[i] = (scale[i] > 0) ? (1 / sqrt(scale[i])) : zero(T)
	end
	return nothing
end

function cache!(cache, refs::MtlVector, interaction::MtlVector, weights::MtlVector, scale::MtlVector, nthreads::Integer)
	nblocks = cld(length(cache[1]), nthreads) 
	Metal.@sync @metal threads=nthreads groups=nblocks cache!_kernel!(cache[1], refs, interaction, weights, scale)
end

function cache!_kernel!(cache, refs, interaction, weights, scale)
	i = thread_position_in_grid_1d()
	if i <= length(cache)
		@inbounds cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
	return nothing
end


end

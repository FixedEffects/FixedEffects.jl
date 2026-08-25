module MetalExt
using FixedEffects, Metal
using FixedEffects: FixedEffectCoefficients, AbstractWeights, UnitWeights, LinearAlgebra, Adjoint, mul!, rmul!, lsmr!, AbstractFixedEffectLinearMap, copy_internal!, AtomicGather, BucketGather
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
	scales::Vector{MtlVector{T}}
	caches::Vector{MtlVector{T}}
	gathers::Vector{Union{AtomicGather, BucketGather}}
end

function _metal_threadgroup_width()
	width = Int(device().maxThreadsPerThreadgroup.width)
	return prevpow(2, width)
end

function bucketize_refs(refs::AbstractVector{<:Integer}, n::Int)
	# count the number of obs per group
	counts = zeros(Int, n)
	@inbounds for r in refs
		counts[r] += 1
	end
	# offsets is vcat(1, cumsum(counts))
	offsets = Vector{Int}(undef, n + 1)
	offsets[1] = 1
	@inbounds for k in 1:n
		offsets[k+1] = offsets[k] + counts[k]
	end

	perm = Vector{Int}(undef, length(refs))
	next = offsets[1:n]
	@inbounds for i in eachindex(refs)
		r = refs[i]
		p = next[r]
		perm[p] = i
		next[r] = p + 1
	end
	return MtlVector{Int}(perm), MtlVector{Int}(offsets)
end

function FixedEffectLinearMapMetal{T}(fes::Vector{<:FixedEffect}) where {T}
	fes2 = [_mtl(T, fe) for fe in fes]
	scales = [Metal.zeros(T, fe.n) for fe in fes]
	caches = [Metal.zeros(T, length(fe.refs)) for fe in fes]
	G = Union{AtomicGather, BucketGather}
	gathers = Vector{G}(undef, length(fes))
	Threads.@threads for i in 1:length(fes)
		refs = fes[i].refs
		n = fes[i].n
		# bucketize (one threadgroup per group) for low cardinality; else atomic adds
		if n < min(100_000, div(length(refs), 16))
			perm, offsets = bucketize_refs(refs, n)
			gathers[i] = BucketGather(perm, offsets)
		else
			gathers[i] = AtomicGather()
		end
	end
	return FixedEffectLinearMapMetal{T}(fes2, scales, caches, gathers)
end

function FixedEffects.gather!(fecoef::MtlVector, refs::MtlVector, α::Number, y::MtlVector, cache::MtlVector, g::BucketGather)
	n = length(fecoef)
	nthreads = _metal_threadgroup_width()
	Metal.@sync @metal threads=nthreads groups=n gather_kernel_bin!(fecoef, refs, α, y, cache, g.perm, g.offsets, Val(nthreads))
end

function FixedEffects.gather!(fecoef::MtlVector, refs::MtlVector, α::Number, y::MtlVector, cache::MtlVector, ::AtomicGather)
	nthreads = _metal_threadgroup_width()
	nblocks = cld(length(y), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks gather_kernel!(fecoef, refs, α, y, cache)
end

function gather_kernel_bin!(fecoef, refs, α, y, cache, perm, offsets, ::Val{NT}) where {NT}
    k   = Int(threadgroup_position_in_grid().x)
    tid = Int(thread_position_in_threadgroup().x)
    nt  = Int(threads_per_threadgroup().x)

    # threadgroup scratch
    T = eltype(fecoef)
    shared = Metal.MtlThreadGroupArray(T, NT)

    start = @inbounds offsets[k]
    stop  = @inbounds offsets[k+1] - 1

    acc = zero(T)

    # each thread walks its portion of the bucket
    j = start + tid - 1
    while j <= stop
        i = @inbounds perm[j]
        @inbounds acc += (α * y[i] * cache[i])
        j += nt
    end

    @inbounds shared[tid] = acc
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    # tree reduction in shared memory
    offset = nt ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds shared[tid] += shared[tid + offset]
        end
        Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        offset ÷= 2
    end

    # one write per coefficient (no atomics needed if groups == n and 1 group per k)
    if tid == 1
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

function FixedEffects.scatter!(y::MtlVector, α::Number, fecoef::MtlVector, refs::MtlVector, cache::MtlVector)
	nthreads = _metal_threadgroup_width()
	nblocks = cld(length(y), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks scatter_kernel!(y, α, fecoef, refs, cache)
end

function FixedEffects.scatter!(y::MtlVector, α::Number, fecoef::MtlVector, refs::MtlVector, cache::MtlVector, β::Number)
	nthreads = _metal_threadgroup_width()
	nblocks = cld(length(y), nthreads)
	if iszero(β)
		Metal.@sync @metal threads=nthreads groups=nblocks scatter_kernel_zero!(y, α, fecoef, refs, cache)
	elseif isone(β)
		Metal.@sync @metal threads=nthreads groups=nblocks scatter_kernel!(y, α, fecoef, refs, cache)
	else
		Metal.@sync @metal threads=nthreads groups=nblocks scatter_kernel_scaled!(y, α, fecoef, refs, cache, β)
	end
end

function scatter_kernel!(y, α, fecoef, refs, cache)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		@inbounds y[i] += α * fecoef[refs[i]] * cache[i]
	end
	return nothing
end

function scatter_kernel_zero!(y, α, fecoef, refs, cache)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		@inbounds y[i] = α * fecoef[refs[i]] * cache[i]
	end
	return nothing
end

function scatter_kernel_scaled!(y, α, fecoef, refs, cache, β)
	i = thread_position_in_grid_1d()
	if i <= length(y)
		@inbounds y[i] = β * y[i] + α * fecoef[refs[i]] * cache[i]
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
	tmp::Vector{T}
	fes::Vector{<:FixedEffect}
end

	
function FixedEffects.AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:Metal}}) where {T}
	T === Float32 || throw(ArgumentError("The Metal backend supports Float32 solves only; pass double_precision=false or use method=:cpu for Float64."))
	m = FixedEffectLinearMapMetal{T}(fes)
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
		scale!(scale, fe.refs, fe.interaction, feM.weights)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, feM.weights, scale)
	end	
	return feM
end

function scale!(scale::MtlVector, refs::MtlVector, interaction::MtlVector, weights::MtlVector)
	nthreads = _metal_threadgroup_width()
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

function cache!(cache::MtlVector, refs::MtlVector, interaction::MtlVector, weights::MtlVector, scale::MtlVector)
	nthreads = _metal_threadgroup_width()
	nblocks = cld(length(cache), nthreads)
	Metal.@sync @metal threads=nthreads groups=nblocks cache!_kernel!(cache, refs, interaction, weights, scale)
end

function cache!_kernel!(cache, refs, interaction, weights, scale)
	i = thread_position_in_grid_1d()
	if i <= length(cache)
		@inbounds cache[i] = interaction[i] * sqrt(weights[i]) * scale[refs[i]]
	end
	return nothing
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

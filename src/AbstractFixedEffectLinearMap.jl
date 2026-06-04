##############################################################################
## 
## 
## Implement AbstractFixedEffectLinearMap
##
## Model matrix of categorical variables
## mutiplied by diag(1/sqrt(∑w * interaction^2, ..., ∑w * interaction^2) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## eltyp
## size
## mul!
##
##############################################################################

abstract type AbstractFixedEffectLinearMap{T} end

# Per-fixed-effect plan for the adjoint gather (A'u). Chosen once at construction and
# dispatched on by each backend's `gather!`. The forward map (A = scatter) needs only the
# preconditioner `caches[i]` and is shared; the gather plan lives in `gathers[i]`.
# CPU uses Serial/Threaded; the GPU backends use Atomic/Bucket.
struct SerialGather end
struct ThreadedGather{V<:AbstractVector}
	buffers::Vector{V}              # one length-fe.n accumulator per thread
	ranges::Vector{UnitRange{Int}}  # contiguous row chunks
end
struct AtomicGather end
struct BucketGather{V<:AbstractVector}
	perm::V        # observation indices sorted by group
	offsets::V     # CSR offsets into perm (length ngroups + 1)
end

Base.adjoint(fem::AbstractFixedEffectLinearMap) = Adjoint(fem)

function Base.size(fem::AbstractFixedEffectLinearMap, dim::Integer)
	(dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

Base.eltype(x::AbstractFixedEffectLinearMap{T}) where {T} = T

function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients, 
	Cfem::Adjoint{T, <:AbstractFixedEffectLinearMap{T}},
	y::AbstractVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (fecoef, fe, cache, gather) in zip(fecoefs.x, fem.fes, fem.caches, fem.gathers)
		gather!(fecoef, fe.refs, α, y, cache, gather)
	end
	return fecoefs
end

function LinearAlgebra.mul!(y::AbstractVector, fem::AbstractFixedEffectLinearMap, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		scatter!(y, α, fecoef, fe.refs, cache)
	end
	return y
end


##############################################################################
##
## FixedEffectCoefficients: the whitened coefficient vector x seen by lsmr!,
## stored as one k × nlevels matrix per AbsorbedBlock (column g holds the
## whitened coordinates of group g).
##
## We define the methods lsmr! needs (duck typing):
## copyto!, fill!, rmul!, axpy!, norm, similar
##
## Do not define iteration on each block since it would conflict with eltype
##
##############################################################################

struct FixedEffectCoefficients{U <: AbstractArray}
	x::Vector{U}
end

Base.eltype(fecoefs::FixedEffectCoefficients{U}) where {U} = eltype(U)
Base.length(fecoefs::FixedEffectCoefficients) = sum(length(x) for x in fecoefs.x)

function LinearAlgebra.norm(fecoefs::FixedEffectCoefficients)
	out = zero(eltype(fecoefs))
	for x in fecoefs.x
		out += sum(abs2, x)
	end
	sqrt(out)
end

function Base.fill!(fecoefs::FixedEffectCoefficients, α::Number)
	for x in fecoefs.x
		fill!(x, eltype(fecoefs)(α))
	end
	return fecoefs
end

function LinearAlgebra.rmul!(fecoefs::FixedEffectCoefficients, α::Number)
	for x in fecoefs.x
		rmul!(x, eltype(fecoefs)(α))
	end
	return fecoefs
end

function Base.copyto!(fecoefs1::FixedEffectCoefficients, fecoefs2::FixedEffectCoefficients)
	for (x1, x2) in zip(fecoefs1.x, fecoefs2.x)
		copyto!(x1, x2)
	end
	return fecoefs1
end

function LinearAlgebra.axpy!(α::Number, fecoefs1::FixedEffectCoefficients, fecoefs2::FixedEffectCoefficients)
	for (x1, x2) in zip(fecoefs1.x, fecoefs2.x)
		axpy!(α, x1, x2)
	end
	return fecoefs2
end

function Base.similar(fecoefs::FixedEffectCoefficients, ::Type{T} = eltype(fecoefs)) where {T}
	return FixedEffectCoefficients([similar(x, T) for x in fecoefs.x])
end

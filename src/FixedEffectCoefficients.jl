# Define methods used in LSMR

##############################################################################
## 
## FixedEffectCoefficients : vector x in A'Ax = A'b
##
## We define these methods used in lsmr! (duck typing): 
## copyto!, fill!, rmul!, axpy!, norm
##
## Do not define iteration on each fixedeffect since it would conflict with eltype
##
##############################################################################

struct FixedEffectCoefficients{U <: AbstractVector}
	x::Vector{U}
end

Base.eltype(fecoef::FixedEffectCoefficients{U}) where {U} = eltype(U)
Base.length(fecoef::FixedEffectCoefficients) = sum(length(x) for x in fecoef.x)
LinearAlgebra.norm(fecoef::FixedEffectCoefficients) = sqrt(sum(sum(abs2, x) for x in fecoef.x))

function Base.fill!(fecoef::FixedEffectCoefficients, α::Number)
	for x in fecoef.x
		fill!(x, α)
	end
	return fecoef
end

function LinearAlgebra.rmul!(fecoef::FixedEffectCoefficients, α::Number)
	for x in fecoef.x
		rmul!(x, α)
	end
	return fecoef
end

function Base.copyto!(fecoef1::FixedEffectCoefficients, fecoef2::FixedEffectCoefficients)
	for (x1, x2) in zip(fecoef1.x, fecoef2.x)
		copyto!(x1, x2)
	end
	return fecoef1
end

function LinearAlgebra.axpy!(α::Number, fecoef1::FixedEffectCoefficients, fecoef2::FixedEffectCoefficients)
	for (x1, x2) in zip(fecoef1.x, fecoef2.x)
		axpy!(α, x1, x2)
	end
	return fecoef2
end





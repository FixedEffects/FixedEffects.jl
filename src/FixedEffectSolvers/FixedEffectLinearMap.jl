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

struct FixedEffectCoefficients{T}
	x::Vector{<:AbstractVector{T}}
end

Base.eltype(fecoef::FixedEffectCoefficients{T}) where {T} = T
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

##############################################################################
## 
## FixedEffectLinearMap
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

struct FixedEffectLinearMap{T}
	fes::Vector{<:FixedEffect}
	sqrtw::AbstractVector{T}
	colnorm::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
end

LinearAlgebra.adjoint(fem::FixedEffectLinearMap) = Adjoint(fem)

function Base.size(fem::FixedEffectLinearMap, dim::Integer)
	(dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

Base.eltype(x::FixedEffectLinearMap{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, fem::FixedEffectLinearMap, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		demean!(y, α, fecoef, fe.refs, cache)
	end
	return y
end

function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients, Cfem::Adjoint{T, FixedEffectLinearMap{T}},
				y::AbstractVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		mean!(fecoef, fe.refs, α, y, cache)
	end
	return fecoefs
end

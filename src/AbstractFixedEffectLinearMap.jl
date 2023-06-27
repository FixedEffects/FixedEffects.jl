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
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		gather!(fecoef, fe.refs, α, y, cache, fem.nthreads)
	end
	return fecoefs
end

function LinearAlgebra.mul!(y::AbstractVector, fem::AbstractFixedEffectLinearMap, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		scatter!(y, α, fecoef, fe.refs, cache, fem.nthreads)
	end
	return y
end


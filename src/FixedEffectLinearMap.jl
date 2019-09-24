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

eltype(fecoef::FixedEffectCoefficients{T}) where {T} = T
length(fecoef::FixedEffectCoefficients) = sum(length(x) for x in fecoef.x)
norm(fecoef::FixedEffectCoefficients) = sqrt(sum(sum(abs2, x) for x in fecoef.x))

function fill!(fecoef::FixedEffectCoefficients, α::Number)
	for x in fecoef.x
		fill!(x, α)
	end
	return fecoef
end

function rmul!(fecoef::FixedEffectCoefficients, α::Number)
	for x in fecoef.x
		rmul!(x, α)
	end
	return fecoef
end

function copyto!(fecoef1::FixedEffectCoefficients, fecoef2::FixedEffectCoefficients)
	for (x1, x2) in zip(fecoef1.x, fecoef2.x)
		copyto!(x1, x2)
	end
	return fecoef1
end

function axpy!(α::Number, fecoef1::FixedEffectCoefficients, fecoef2::FixedEffectCoefficients)
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


function FixedEffectLinearMap{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}}) where {T}
	sqrtw = convert(AbstractVector{T}, sqrtw)
	colnorm = [colnorm!(zeros(T, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes]
	caches = [cache!(zeros(T, length(sqrtw)), fe.interaction, sqrtw, scale, fe.refs) for (fe, scale) in zip(fes, colnorm)]
	return FixedEffectLinearMap{T}(fes, sqrtw, colnorm, caches)
end

function colnorm!(fecoef::AbstractVector, refs::AbstractVector, interaction::AbstractVector, sqrtw::AbstractVector)
	@inbounds @simd for i in eachindex(refs)
		fecoef[refs[i]] += abs2(interaction[i] * sqrtw[i])
	end
	fecoef .= sqrt.(fecoef)
end

function cache!(y::AbstractVector, interaction::AbstractVector, sqrtw::AbstractVector, fecoef::AbstractVector, refs::AbstractVector)
	@inbounds @simd for i in eachindex(y)
		y[i] = interaction[i] * sqrtw[i] / fecoef[refs[i]]
	end
	return y
end

adjoint(fem::FixedEffectLinearMap) = Adjoint(fem)

function size(fem::FixedEffectLinearMap, dim::Integer)
	(dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end
eltype(x::FixedEffectLinearMap{T}) where {T} = T

function mul!(y::AbstractVector, fem::FixedEffectLinearMap, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		demean!(y, α, fecoef, fe.refs, cache)
	end
	return y
end

function demean!(y::AbstractVector, α::Number, fecoef::AbstractVector, refs::AbstractVector, cache::AbstractVector)
	@simd ivdep for i in eachindex(y)
		@inbounds y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

function mul!(fecoefs::FixedEffectCoefficients, Cfem::Adjoint{T, FixedEffectLinearMap{T}},
				y::AbstractVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		mean!(fecoef, fe.refs, α, y, cache)
	end
	return fecoefs
end

function mean!(fecoef::AbstractVector, refs::AbstractVector, α::Number, y::AbstractVector, cache::AbstractVector)
	@simd ivdep for i in eachindex(y)
		@inbounds fecoef[refs[i]] += α * y[i] * cache[i]
	end
end




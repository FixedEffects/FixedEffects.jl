##############################################################################
##
## AbstractFixedEffectLinearMap
##
## The whitened operator A = W^(1/2) * (fixed-effect design) * R used by lsmr!,
## where R is the per-group block transform stored in the AbsorptionPlan.
##
## Each backend stores an AbsorptionPlan in a `plan` field and defines mul!
## for itself and its adjoint. lsmr! needs (duck typing): eltype, size, mul!.
##
##############################################################################

abstract type AbstractFixedEffectLinearMap{T} end

Base.adjoint(fem::AbstractFixedEffectLinearMap) = Adjoint(fem)

function Base.size(fem::AbstractFixedEffectLinearMap, dim::Integer)
	(dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? _ncoef(fem.plan) : 1
end

Base.eltype(x::AbstractFixedEffectLinearMap{T}) where {T} = T

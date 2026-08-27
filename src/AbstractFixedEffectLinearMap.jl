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

# Concrete subtypes must be mutable and expose two fields used by shared code:
#   fes  — the original fixed effects, used for the observation count and coefficient recovery;
#   plan — the AbsorptionPlan used by the operator, replaceable when weights change.
abstract type AbstractFixedEffectLinearMap{T} end

Base.eltype(x::AbstractFixedEffectLinearMap{T}) where {T} = T

Base.adjoint(fem::AbstractFixedEffectLinearMap) = Adjoint(fem)

function Base.size(fem::AbstractFixedEffectLinearMap, dim::Integer)
	if dim == 1
		return length(fem.fes[1].refs)
	elseif dim == 2
		return sum(block_width(block) * block.n for block in fem.plan.blocks)
	else
		1
	end
end

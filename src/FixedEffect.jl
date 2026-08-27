##############################################################################
##
## FixedEffect
##
##############################################################################

struct FixedEffect{R <: AbstractVector{<:Integer}, I <: AbstractVector{<:Real}}
	refs::R                 # group of each observation, in 1:n (0 marks a missing group; such rows must be dropped before solving)
	interaction::I          # the continuous interaction
	n::Int                  # Number of potential values (= maximum(refs))
	function FixedEffect{R, I}(refs, interaction, n) where {R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
		length(refs) == length(interaction) || throw(DimensionMismatch(
			"cannot match refs of length $(length(refs)) with interaction of length $(length(interaction))"))
		return new(refs, interaction, n)
	end
end

function FixedEffect(args...; interaction::AbstractVector = uweights(length(args[1])))
	g = GroupedArray(args..., sort = nothing)
	# Store refs as Int32 (refs lie in [0, ngroups]) to halve the dominant memory stream read by the
	# scatter/gather kernels on every solver iteration. GroupedArrays always builds Int64 groups (it
	# needs signed sentinels during construction); narrowing here, where FixedEffect manufactures its
	# own ref representation, lets every backend (CPU/GPU) and solve_coefficients! stream the smaller
	# type. The rare ngroups > typemax(Int32) keeps the original integer type.
	refs = g.ngroups > typemax(Int32) ? g.groups : convert(Vector{Int32}, g.groups)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, g.ngroups)
end

Base.show(io::IO, ::FixedEffect) = print(io, "Fixed Effects")

function Base.show(io::IO, ::MIME"text/plain", fe::FixedEffect)
	print(io, fe, ':')
	print(io, "\n  refs (", length(fe.refs), "-element ", typeof(fe.refs), "):")
	print(io, "\n    [", string.(Int.(fe.refs[1:min(5, length(fe.refs))])).*", "..., "... ]")
	if fe.interaction isa UnitWeights
		print(io, "\n  interaction (UnitWeights):")
		print(io, "\n    none")
	else
		print(io, "\n  interaction (", length(fe.interaction), "-element ", typeof(fe.interaction), "):")
		print(io, "\n    [", (sprint(show, x; context=:compact=>true)*", " for x in fe.interaction[1:min(5, length(fe.interaction))])..., "... ]")
	end
end

Base.size(fe::FixedEffect) = size(fe.refs)
Base.length(fe::FixedEffect) = length(fe.refs)
Base.eltype(::FixedEffect{R,I}) where {R,I} = eltype(I)

Base.getindex(fe::FixedEffect, ::Colon) = fe

@propagate_inbounds function Base.getindex(fe::FixedEffect, esample)
	@boundscheck checkbounds(fe.refs, esample)
	@boundscheck checkbounds(fe.interaction, esample)
	@inbounds refs = fe.refs[esample]
	@inbounds interaction = fe.interaction[esample]
	return FixedEffect{typeof(fe.refs), typeof(fe.interaction)}(refs, interaction, fe.n)
end


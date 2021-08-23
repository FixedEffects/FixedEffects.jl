##############################################################################
##
## FixedEffect
##
##############################################################################

struct FixedEffect{R <: AbstractVector{<:Integer}, I <: AbstractVector{<:Real}}
	refs::R                 # refs must be between 0 and n
	interaction::I          # the continuous interaction
	n::Int                  # Number of potential values (= maximum(refs))
	function FixedEffect{R, I}(refs, interaction, n) where {R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
		length(refs) == length(interaction) || throw(DimensionMismatch(
			"cannot match refs of length $(length(refs)) with interaction of length $(length(interaction))"))
		return new(refs, interaction, n)
	end
end

function FixedEffect(args...; interaction::AbstractVector = uweights(length(args[1])))
	g = GroupedArray(args...)
	FixedEffect{typeof(g.groups), typeof(interaction)}(g.groups, interaction, g.ngroups)
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
##############################################################################
##
## Find connected components
## 
##############################################################################
# Return a vector of sets that contains the indices of each unique value
function refsrev(fe::FixedEffect)
	out = Vector{Int}[Int[] for _ in 1:fe.n]
	for i in eachindex(fe.refs)
		push!(out[fe.refs[i]], i)
	end
	return out
end

# Returns a vector of all components
# A component is a vector that, for each fixed effect, 
# contains all the refs that are included in the component.
function components(fes::AbstractVector{<:FixedEffect})
	refs_vec = Vector{Int}[fe.refs for fe in fes]
	refsrev_vec = Vector{Vector{Int}}[refsrev(fe) for fe in fes]
	visited = falses(length(refs_vec[1]))
	out = Vector{Set{Int}}[]
	for i in eachindex(visited)
		if !visited[i]
			# obs not visited yet, so create new component
			component_vec = Set{Int}[Set{Int}() for _ in 1:length(refsrev_vec)]
			# visit all obs in the same components
			tovisit = Set{Int}(i)
			while !isempty(tovisit)
				for (component, refs, refsrev) in zip(component_vec, refs_vec, refsrev_vec)
					ref = refs[i]
					# if group is not in component yet
					if ref ∉ component
						# add group to the component
						push!(component, ref)
						# visit other observations in same group
						union!(tovisit, refsrev[ref])
					end
				end
				# mark obs as visited
				i = pop!(tovisit)
				visited[i] = true
			end            
			push!(out, component_vec)
		end
	end
	return out
end

##############################################################################
##
## normalize! a vector of fixedeffect coefficients using connected components
## 
##############################################################################

function normalize!(fecoefs::AbstractVector{<: Vector{<: Real}}, fes::AbstractVector{<:FixedEffect}; kwargs...)
	# The solution is generally not unique. Find connected components and scale accordingly
	idx = findall(fe -> isa(fe.interaction, UnitWeights), fes)
	length(idx) >= 2 && rescale!(view(fecoefs, idx), view(fes, idx))
	return fecoefs
end

function rescale!(fecoefs::AbstractVector{<: Vector{<: Real}}, fes::AbstractVector{<:FixedEffect})
	for component_vec in components(fes)
		m = 0.0
		# demean all fixed effects except the first
		for j in length(fecoefs):(-1):2
			fecoef, component = fecoefs[j], component_vec[j]
			mj = 0.0
			for k in component
				mj += fecoef[k]
			end
			mj = mj / length(component)
			for k in component
				fecoef[k] -= mj
			end
			m += mj
		end
		# rescale the first fixed effects
		fecoef, component = fecoefs[1], component_vec[1]
		for k in component
			fecoef[k] += m
		end
	end
end

function full(fecoefs::AbstractVector{<: Vector{<: Real}}, fes::AbstractVector{<:FixedEffect})
	[fecoef[fe.refs] for (fecoef, fe) in zip(fecoefs, fes)]
end

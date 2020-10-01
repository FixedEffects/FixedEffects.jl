##############################################################################
##
## FixedEffect
##
## The categoricalarray may have pools that are never referred. Note that the pool does not appear in FixedEffect anyway.
##
##############################################################################

struct FixedEffect{R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
	refs::R                 # refs must be between 0 and n
	interaction::I          # the continuous interaction
	n::Int                  # Number of potential values (= maximum(refs))
	function FixedEffect{R, I}(refs, interaction, n) where {R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
		length(refs) == length(interaction) || error("refs and interaction don't have the same length")
		new(refs, interaction, n)
	end
end

function FixedEffect(args...; interaction::AbstractVector = uweights(length(args[1])))
	g = group(args...)
	FixedEffect{typeof(g.refs), typeof(interaction)}(g.refs, interaction, g.n)
end

function Base.show(io::IO, fe::FixedEffect)
	println(io, "Refs:        ", length(fe.refs), "-element ", typeof(fe.refs))
	println(io, "             ", Int.(fe.refs[1:min(5, length(fe.refs))]), "...")
	println(io, "Interaction: ", length(fe.interaction), "-element ", typeof(fe.interaction))
	println(io, "             ", fe.interaction[1:min(5, length(fe.interaction))], "...")
end

Base.length(fe::FixedEffect) = length(fe.refs)
Base.eltype(fe::FixedEffect) = eltype(I)

Base.getindex(fe::FixedEffect, esample::Colon) = fe
function Base.getindex(fe::FixedEffect{R, I}, esample::AbstractVector) where {R, I}
	if fe.interaction isa UnitWeights
		FixedEffect{R, I}(fe.refs[esample], uweights(eltype(I), sum(esample)),  fe.n)
	else
		FixedEffect{R, I}(fe.refs[esample], fe.interaction[esample],  fe.n)
	end
end


##############################################################################
##
## group combines multiple refs
## Missings have a ref of 0
## 
##############################################################################

mutable struct GroupedArray{N} <: AbstractArray{UInt32, N}
	refs::Array{UInt32, N}   # refs must be between 0 and n. 0 means missing
	n::Int                   # Number of potential values (= maximum(refs))
end
group(xs::GroupedArray) = xs

function group(xs::AbstractArray)
	refs = Array{UInt32}(undef, size(xs))
	invpool = Dict{eltype(xs), UInt32}()
	n = 0
	has_missing = false
	@inbounds for i in eachindex(xs)
		x = xs[i]
		if x === missing
			refs[i] = 0
			has_missing = true
		else
			lbl = get(invpool, x, 0)
			if lbl !== 0
				refs[i] = lbl
			else
				n += 1
				refs[i] = n
				invpool[x] = n
			end
		end
	end
	return GroupedArray{ndims(xs)}(refs, n)
end

function group(args...)
	g1 = deepcopy(group(args[1]))
	for j = 2:length(args)
		gj = group(args[j])
		length(g1.refs) == length(gj.refs) || throw(DimensionError())
		combine!(g1, gj)
	end
	factorize!(g1)
end

function combine!(g1::GroupedArray, g2::GroupedArray)
	@inbounds for i in eachindex(g1.refs, g2.refs)
		# if previous one is missing or this one is missing, set to missing
		g1.refs[i] = (g1.refs[i] == 0 || g2.refs[i] == 0) ? 0 : g1.refs[i] + (g2.refs[i] - 1) * g1.n
	end
	g1.n = g1.n * g2.n
	return g1
end

function factorize!(x::GroupedArray{N}) where {N}
	refs = x.refs
	uu = sort!(unique(refs))
	has_missing = uu[1] == 0
	ngroups = length(uu) - has_missing
	dict = Dict{UInt32, UInt32}(zip(uu, UInt32(1-has_missing):UInt32(ngroups)))
	for i in eachindex(refs)
		refs[i] = dict[refs[i]]
	end
	GroupedArray{N}(refs, ngroups)
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
	refs_vec = Vector{UInt32}[fe.refs for fe in fes]
	refsrev_vec = Vector{Vector{Int}}[refsrev(fe) for fe in fes]
	visited = falses(length(refs_vec[1]))
	out = Vector{Set{Int}}[]
	for i in eachindex(visited)
		if !visited[i]
			# obs not visited yet, so create new component
			component_vec = Set{UInt32}[Set{UInt32}() for _ in 1:length(refsrev_vec)]
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
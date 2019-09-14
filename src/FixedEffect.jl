##############################################################################
##
## FixedEffect
##
## The categoricalarray may have pools that are never referred. Note that the pool does not appear in FixedEffect anyway.
##
##############################################################################

struct FixedEffect{R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
    refs::R                 # refs of the original CategoricalVector
    interaction::I          # the continuous interaction
    n::Int                  # Number of potential values (= maximum(refs))
    function FixedEffect{R, I}(refs, interaction, n) where {R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
        maximum(refs) > n && error("Categorical Vector used to construct Fixed Effect is malformed. Some elements of refs do not refer to pool")
        new(refs, interaction, n)
    end
end

function FixedEffect(args...; interaction::AbstractVector = Ones{Float64}(length(args[1])))
    if length(args) != 1
        FixedEffect(group(args...); interaction = interaction)
    else
        x = args[1]
        if !isa(x, CategoricalVector)
            FixedEffect(categorical(x); interaction = interaction)
        else
            FixedEffect{typeof(x.refs), typeof(interaction)}(x.refs, interaction, length(x.pool))
        end
    end
end

function Base.show(io::IO, fe::FixedEffect)
    println(io, "Refs:        ", length(fe.refs), "-element ", typeof(fe.refs))
    println(io, "             ", Int.(fe.refs[1:min(5, length(fe.refs))]), "...")
    println(io, "Interaction: ", length(fe.interaction), "-element ", typeof(fe.interaction))
    println(io, "             ", fe.interaction[1:min(5, length(fe.interaction))], "...")
end

Base.ismissing(fe::FixedEffect) = any(fe.refs .== 0)  | ismissing(fe.interaction)
Base.length(fe::FixedEffect) = length(fe.refs)

##############################################################################
##
## group transform multiple CategoricalVector into one
## Output is a CategoricalVector where pool is type Int64, equal to ranking of group
## Missing in some row mean result has Missing on this row
## 
##############################################################################

function group(args...)
    v = categorical(args[1])
    if length(args) == 1
        refs = v.refs
    else
        refs, ngroups = convert(Vector{UInt}, v.refs), length(levels(v))
        for j = 2:length(args)
            v = categorical(args[j])
            refs, ngroups = pool_combine!(refs, v, ngroups)
        end
    end
    factorize!(refs)
end
function pool_combine!(refs::Array{T, N}, dv::CategoricalVector, ngroups::Integer) where {T, N}
    for i in 1:length(refs)
        # if previous one is NA or this one is NA, set to NA
        refs[i] = (dv.refs[i] == 0 || refs[i] == zero(T)) ? zero(T) : refs[i] + (dv.refs[i] - 1) * ngroups
    end
    return refs, ngroups * length(levels(dv))
end

function factorize!(refs::Vector{T}) where {T}
    uu = sort!(unique(refs))
    has_missing = uu[1] == 0
    dict = Dict{T, Int}(zip(uu, (1-has_missing):(length(uu)-has_missing)))
    newrefs = zeros(UInt32, length(refs))
    for i in 1:length(refs)
         newrefs[i] = dict[refs[i]]
    end
    Tout = has_missing ? Union{Int, Missing} : Int
    CategoricalArray{Tout, 1}(newrefs, CategoricalPool(collect(1:(length(uu)-has_missing))))
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

## Connected component : Breadth-first search
## Returns a vector of all components
## A component is a vector that, for each fixed effect, has all the refs that are included in id.
function components(fes::AbstractVector{<:FixedEffect})
    refs_vec = Vector{UInt32}[fe.refs for fe in fes]
    refsrev_vec = Vector{Vector{Int}}[refsrev(fe) for fe in fes]
    visited = falses(length(refs_vec[1]))
    out = Vector{Set{Int}}[]
    for i in eachindex(visited)
        if !visited[i]
            # create new component
            component_vec = Set{Int}[Set{Int}() for _ in 1:length(refsrev_vec)]
            # find all elements of this new component
            tovisit = Set{Int}(i)
            while !isempty(tovisit)
                i = pop!(tovisit)
                # mark index as visited
                visited[i] = true
                for (component, refs, refsrev) in zip(component_vec, refs_vec, refsrev_vec)
                    # if group is not in component yet
                    if !(refs[i] in component)
                        # mark group as encountered
                        push!(component, refs[i])
                        # visit other observations in same group
                        # if it has not been visited yet (otherwise i is going to be on the list)
                        for k in refsrev[refs[i]]
                            if !visited[k]
                                push!(tovisit, k)
                            end
                        end
                    end
                end
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
    idx_intercept = findall(fe -> isa(fe.interaction, Ones), fes)
    if length(idx_intercept) >= 2
        rescale!(view(fecoefs, idx_intercept), view(fes, idx_intercept))
    end
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
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
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
    if length(fe.refs) <= 5
        println(io, "             ", convert(Vector{Int}, fe.refs))
    else
        println(io, "             ", convert(Vector{Int}, fe.refs[1:5]), "...")
    end
    println(io, "Interaction: ", length(fe.interaction), "-element ", typeof(fe.interaction))
    if length(fe.interaction) <= 5
        println(io, "             ", fe.interaction, "...")
    else
        println(io, "             ", fe.interaction[1:5], "...")
    end
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
        x = v.refs
    else
        x, ngroups = convert(Vector{UInt}, v.refs), length(levels(v))
        for j = 2:length(args)
            v = categorical(args[j])
            x, ngroups = pool_combine!(x, v, ngroups)
        end
    end
    factorize!(x)
end

#  drop unused levels
function factorize!(refs::Vector{T}) where {T}
    uu = unique(refs)
    sort!(uu)
    has_missing = uu[1] == 0
    dict = Dict{T, Int}(zip(uu, (1-has_missing):(length(uu)-has_missing)))
    newrefs = zeros(UInt32, length(refs))
    for i in 1:length(refs)
         newrefs[i] = dict[refs[i]]
    end
    if has_missing
        Tout = Union{Int, Missing}
    else
        Tout = Int
    end
    CategoricalArray{Tout, 1}(newrefs, CategoricalPool(collect(1:(length(uu)-has_missing))))
end
function pool_combine!(x::Array{T, N}, dv::CategoricalVector, ngroups::Integer) where {T, N}
    for i in 1:length(x)
        # if previous one is NA or this one is NA, set to NA
        x[i] = (dv.refs[i] == 0 || x[i] == zero(T)) ? zero(T) : x[i] + (dv.refs[i] - 1) * ngroups
    end
    return x, ngroups * length(levels(dv))
end

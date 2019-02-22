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
    x = group(args...)
    FixedEffect{typeof(x.refs), typeof(interaction)}(x.refs, interaction, length(x.pool))
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
    length(args) == 1 && return v
    refs, ngroups = convert(Vector{UInt32}, v.refs), length(levels(v))
    for j = 2:length(args)
        v = categorical(args[j])
        refs, ngroups = pool_combine!(refs, v, ngroups)
    end
    Tout = any(refs .== 0) ? Union{Int, Missing} : Int
    return CategoricalArray{Tout, 1}(refs, CategoricalPool(collect(1:ngroups)))
end

function pool_combine!(refs::Array{T, N}, dv::CategoricalVector, ngroups::Integer) where {T, N}
    for i in 1:length(refs)
        # if previous one is NA or this one is NA, set to NA
        refs[i] = (dv.refs[i] == 0 || refs[i] == zero(T)) ? zero(T) : refs[i] + (dv.refs[i] - 1) * ngroups
    end
    return refs, ngroups * length(levels(dv))
end

##############################################################################
##
## FixedEffect
##
## The categoricalarray may have pools that are never referred. Note that the pool does not appear in FixedEffect anyway.
##
##############################################################################

struct FixedEffect{R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
    refs::R                 # refs
    n::Int                  # Number of potential values not including missing (= maximum(refs))
    interaction::I          # the continuous interaction
    function FixedEffect{R, I}(refs, n, interaction) where {R <: AbstractVector{<:Integer}, I <: AbstractVector{<: Real}}
        new(refs, n, interaction)
    end
end

function FixedEffect(args...; interaction::AbstractVector = Ones{Float64}(length(args[1])))
    groups = Vector{Int}(undef, length(args[1]))    
    ngroups, rhashes, gslots, sorted = DataFrames.row_group_slots(args, Val(true), groups, true)
    FixedEffect{Vector{Int}, typeof(interaction)}(groups .-1, ngroups - 1, interaction)
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


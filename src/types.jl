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



# from Pkg.jl
Base.@kwdef mutable struct MiniProgressBar
    max::Int = 1
    header::String = ""
    color::Symbol = :white
    width::Int = 32
    current::Int = 0
    prev::Int = 0
    has_shown::Bool = false
    time_shown::Float64 = 0.0
    percentage::Bool = true
    indent::Int = 4
end

const NONINTERACTIVE_TIME_GRANULARITY = Ref(2.0)
const PROGRESS_BAR_PERCENTAGE_GRANULARITY = Ref(0.1)

function showprogress(io::IO, p::MiniProgressBar)
    if p.max == 0
        perc = 0.0
        prev_perc = 0.0
    else
        perc = p.current / p.max * 100
        prev_perc = p.prev / p.max * 100
    end
    if !isinteractive()
        t = time()
        if p.has_shown && (t - p.time_shown) < NONINTERACTIVE_TIME_GRANULARITY[]
            return
        end
        p.time_shown = t
    end
    p.prev = p.current
    p.has_shown = true
    n_filled = ceil(Int, p.width * perc / 100)
    n_left = p.width - n_filled
    print(io, " "^p.indent)
    printstyled(io, p.header, color=p.color, bold=true)
    print(io, " [")
    print(io, "="^n_filled, ">")
    print(io, " "^n_left, "]  ", )
    if p.percentage
        @printf io "%2.1f %%" perc
    else
        print(io, p.current, "/",  p.max)
    end
    print(io, "\r")
end

function end_progress(io, p::MiniProgressBar)
    ansi_enablecursor = "\e[?25h"
    ansi_clearline = "\e[2K"
    print(io, ansi_enablecursor)
    print(io, ansi_clearline)
end
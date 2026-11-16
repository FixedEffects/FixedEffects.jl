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
    # Render into a buffer, then write it with invokelatest, which is not
    # inferred through: printing to stdout, which is only known to be an IO,
    # compiles calls that packages adding IO methods (e.g. FilePathsBase)
    # invalidate, along with solve_residuals! and its callers
    buf = IOContext(IOBuffer(), :color => get(io, :color, false))
    print(buf, " "^p.indent)
    printstyled(buf, p.header, color=p.color, bold=true)
    print(buf, " [")
    print(buf, "="^n_filled, ">")
    print(buf, " "^n_left, "]  ", )
    if p.percentage
        @printf buf "%2.1f %%" perc
    else
        print(buf, p.current, "/",  p.max)
    end
    print(buf, "\r")
    Base.invokelatest(write, io, take!(buf.io))
    return nothing
end

function end_progress(io::IO, p::MiniProgressBar)
    ansi_enablecursor = "\e[?25h"
    ansi_clearline = "\e[2K"
    Base.invokelatest(print, io, ansi_enablecursor * ansi_clearline)
    return nothing
end
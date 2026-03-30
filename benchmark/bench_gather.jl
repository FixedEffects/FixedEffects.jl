using Random, BenchmarkTools, Base.Threads
println("Julia ", VERSION, " — ", nthreads(), " threads")
Random.seed!(1234)

##############################################################################
# Current serial gather (baseline)
##############################################################################
function gather_serial!(fecoef, refs, α, y, cache)
    @fastmath @inbounds @simd for i in eachindex(y)
        fecoef[refs[i]] += α * y[i] * cache[i]
    end
end

##############################################################################
# Approach 1: CSC-style transposed gather
# Precompute a CSC structure: for each group k, store obs indices.
# Each group k can be processed independently → trivially parallel, zero conflicts.
##############################################################################
struct CSCIndex
    offsets::Vector{Int}
    indices::Vector{Int}
end

function build_csc(refs::AbstractVector{<:Integer}, n::Int)
    N = length(refs)
    counts = zeros(Int, n)
    @inbounds for i in 1:N
        counts[refs[i]] += 1
    end
    offsets = Vector{Int}(undef, n + 1)
    offsets[1] = 1
    @inbounds for k in 1:n
        offsets[k+1] = offsets[k] + counts[k]
    end
    indices = Vector{Int}(undef, N)
    fill!(counts, 0)
    @inbounds for i in 1:N
        k = refs[i]
        counts[k] += 1
        indices[offsets[k] + counts[k] - 1] = i
    end
    return CSCIndex(offsets, indices)
end

function gather_csc_parallel!(fecoef::AbstractVector{T}, csc::CSCIndex, α, y, cache) where T
    offsets, indices = csc.offsets, csc.indices
    n = length(fecoef)
    Threads.@threads for k in 1:n
        s = zero(T)
        @fastmath @inbounds for j in offsets[k]:(offsets[k+1]-1)
            i = indices[j]
            s += y[i] * cache[i]
        end
        @inbounds fecoef[k] += α * s
    end
end

function gather_csc_serial!(fecoef::AbstractVector{T}, csc::CSCIndex, α, y, cache) where T
    offsets, indices = csc.offsets, csc.indices
    n = length(fecoef)
    for k in 1:n
        s = zero(T)
        @fastmath @inbounds for j in offsets[k]:(offsets[k+1]-1)
            i = indices[j]
            s += y[i] * cache[i]
        end
        @inbounds fecoef[k] += α * s
    end
end

##############################################################################
# Approach 2: Per-thread accumulators with manual chunking (@spawn)
##############################################################################
struct PerThreadBuffers{T}
    buffers::Vector{Vector{T}}
end
PerThreadBuffers{T}(n::Int, nt::Int) where T = PerThreadBuffers([zeros(T, n) for _ in 1:nt])

function gather_perthread!(fecoef::AbstractVector{T}, refs, α, y, cache, ptb::PerThreadBuffers{T}) where T
    nt = length(ptb.buffers)
    N = length(y)
    for buf in ptb.buffers
        fill!(buf, zero(T))
    end
    chunk = cld(N, nt)
    @sync for t in 1:nt
        Threads.@spawn begin
            buf = ptb.buffers[t]
            lo = (t-1)*chunk + 1
            hi = min(t*chunk, N)
            @fastmath @inbounds for i in lo:hi
                buf[refs[i]] += y[i] * cache[i]
            end
        end
    end
    @inbounds for buf in ptb.buffers
        @simd for k in eachindex(fecoef)
            fecoef[k] += α * buf[k]
        end
    end
end

##############################################################################
# Benchmarks
##############################################################################
function run_bench(label, N, n_groups)
    println("\n", "="^60)
    println("$label: N=$N, n_groups=$n_groups (avg group size=$(N÷n_groups))")
    println("="^60)

    refs = rand(1:n_groups, N)
    y = rand(N)
    cache = rand(N)
    α = 1.0
    nt = nthreads()

    csc = build_csc(refs, n_groups)
    ptb = PerThreadBuffers{Float64}(n_groups, nt)

    # Verify correctness
    out_ref = zeros(n_groups)
    gather_serial!(out_ref, refs, α, y, cache)

    for (name, fn!) in [
        ("CSC parallel", (out) -> gather_csc_parallel!(out, csc, α, y, cache)),
        ("CSC serial", (out) -> gather_csc_serial!(out, csc, α, y, cache)),
        ("Per-thread chunked", (out) -> gather_perthread!(out, refs, α, y, cache, ptb)),
    ]
        out_test = zeros(n_groups)
        fn!(out_test)
        if !isapprox(out_test, out_ref, rtol=1e-10)
            println("  WARNING $name: INCORRECT (max diff = $(maximum(abs.(out_test .- out_ref))))")
        end
    end

    print("  Serial (baseline):  ")
    b0 = @benchmark gather_serial!(out, $refs, $α, $y, $cache) setup=(out=zeros($n_groups)) evals=1 samples=30
    show(stdout, MIME("text/plain"), b0); println()

    print("  CSC serial:         ")
    b1 = @benchmark gather_csc_serial!(out, $csc, $α, $y, $cache) setup=(out=zeros($n_groups)) evals=1 samples=30
    show(stdout, MIME("text/plain"), b1); println()

    print("  CSC parallel:       ")
    b2 = @benchmark gather_csc_parallel!(out, $csc, $α, $y, $cache) setup=(out=zeros($n_groups)) evals=1 samples=30
    show(stdout, MIME("text/plain"), b2); println()

    print("  Per-thread chunked: ")
    b3 = @benchmark gather_perthread!(out, $refs, $α, $y, $cache, $ptb) setup=(out=zeros($n_groups)) evals=1 samples=30
    show(stdout, MIME("text/plain"), b3); println()

    t0 = median(b0).time
    println("\n  Speedups vs serial baseline:")
    println("    CSC serial:         $(round(t0/median(b1).time, digits=2))x")
    println("    CSC parallel:       $(round(t0/median(b2).time, digits=2))x")
    println("    Per-thread chunked: $(round(t0/median(b3).time, digits=2))x")
end

# Scenario 1: Few large groups (like year FE)
run_bench("Few large groups", 10_000_000, 100)

# Scenario 2: Many medium groups
run_bench("Many medium groups", 10_000_000, 100_000)

# Scenario 3: Many small groups (worker FE)
run_bench("Many small groups (worker FE)", 800_000, 400_000)

# Scenario 4: Moderate groups (firm FE)
run_bench("Moderate groups (firm FE)", 800_000, 50_000)

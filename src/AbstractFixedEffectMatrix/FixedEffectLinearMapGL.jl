using OpenCL, LinearAlgebra

##############################################################################
##
## ClArray
##
##############################################################################

mutable struct CLArray{T, N} <: AbstractArray{T, N}
    ctx::cl.Context
    queue::cl.CmdQueue
    buffer::cl.Buffer{T}
    size::NTuple{N, Int}
end
CLVector{T} = CLArray{T, 1}
function CLArray(buf::cl.Buffer, queue::cl.CmdQueue, sz::Tuple{Vararg{Int}})
    ctx = cl.context(buf)
    CLArray(ctx, queue, buf, sz)
end
function CLArray(queue::cl.CmdQueue,
                 flags::Tuple{Vararg{Symbol}},
                 hostarray::AbstractArray{T,N}) where {T, N}
    ctx = cl.context(queue)
    buf = cl.Buffer(T, ctx, flags, hostbuf=hostarray)
    sz = size(hostarray)
    CLArray(ctx, queue, buf, sz)
end
function CLArray(queue::cl.CmdQueue, hostarray::AbstractArray; flags=(:rw, :copy))
  CLArray(queue, (:rw, :copy), hostarray)
end

function Base.fill!(x::ClArray{T},  x::T) where T
  v = cl.opencl_version(x.ctx)
  if v.major == 1 && v.minor >= 2
      cl.fill!(cl.queue, cl.buffer, x)
  else
      buf = cl.Buffer(T, ctx, (:rw, :copy), prod(dims), hostbuf=fill(x, dims))
  end
end




function Base.fill(::Type{T}, queue::cl.CmdQueue, x::T, dims...) where T
    ctx = cl.info(queue, :context)
    v = cl.opencl_version(ctx)
    if v.major == 1 && v.minor >= 2
        buf = cl.Buffer(T, ctx, prod(dims))
        cl.fill!(queue, buf, x)
    else
        buf = cl.Buffer(T, ctx, (:rw, :copy), prod(dims), hostbuf=fill(x, dims))
    end
    return CLArray(buf, queue, dims)
end
cl.buffer(A::CLArray) = A.buffer
Base.size(A::CLArray) = A.size
function Base.show(io::IO, A::CLArray{T,N}) where {T, N}
  print(io, "CLArray{$T,$N}($(cl.buffer(A)),$(size(A)))")
end
clazeros(::Type{T}, queue::cl.CmdQueue, n::Integer) where {T} = fill(T, queue, T(0), n::Integer)
claones(::Type{T}, queue::cl.CmdQueue, n::Integer) where {T} = fill(T, queue, T(1), n::Integer)
function cla(T::Type, queue::cl.CmdQueue, x::AbstractVector)
  CLArray(queue, convert(Vector{T}, x))
end

function Base.copyto!(dest::Vector, src::CLVector)
      copy!(src.queue, dest, cl.buffer(src))
      return dest
end
function Base.copyto!(dest::CLVector, src::Vector)
      copy!(dest.queue, cl.buffer(dest), src)
      return dest
end




    ##############################################################################
    ##
    ## Kernel
    ##
    ##############################################################################


    const mean_kernel = "
    inline void atomicAdd_f(__global float* address, float value)
      {
        float old = value;
        while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
      }
       __kernel void mean_kernel(__global float *fecoef,
                         __global const unsigned int *refs,
                         __global const float *y,
                         float alpha,
                         __global const float *cache)
        {
          int gid = get_global_id(0);
          atomicAdd_f(&fecoef[refs[gid]], alpha * y[gid] * cache[gid]);
        }
    "

    const demean_kernel = "
       __kernel void demean_kernel(__global float *y,
                         __global const float *fecoef,
                         __global const unsigned int *refs,
                          float alpha,
                          __global const float *cache)
        {
          int gid = get_global_id(0);
          y[gid] = y[gid] + fecoef[refs[gid]] * alpha * cache[gid];
        }
    "
    device, ctx, queue = cl.create_compute_context()

    p_demean = cl.Program(ctx, source=demean_kernel) |> cl.build!
    p_mean = cl.Program(ctx, source=mean_kernel) |> cl.build!



function mean!(fecoef::CLVector, refs::CLVector, y::CLVector, alpha::Number, cache::CLVector)
  k = cl.Kernel(p_mean, "mean_kernel")
  queue(k, size(y), nothing, cl.buffer(fecoef), cl.buffer(refs), cl.buffer(y), alpha, cl.buffer(cache))    
end

function demean!(y::CLVector, fecoef::CLVector, refs::CLVector, alpha, cache::CLVector)
  k = cl.Kernel(p_demean, "demean_kernel")
  queue(k, size(y), nothing, cl.buffer(y), cl.buffer(fecoef), cl.buffer(refs), alpha, cl.buffer(cache))
end












##############################################################################
##
## try
##
##############################################################################

using FixedEffects
function cla(T::Type, queue::cl.CmdQueue, fe::FixedEffect)
  refs = cla(UInt32, queue, fe.refs)
  interaction = cla(T, queue, fe.interaction)
  FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end


struct FixedEffectLSMRCL{T} <: AbstractFixedEffectMatrix{T}
  m::FixedEffects.FixedEffectLSMR{T}
  tmp::Vector{T} # used to convert AbstractVector to Vector{T}
  fes::Vector{<:FixedEffect}
end

function AbstractFixedEffectMatrix{T}(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_cl}}) where {T}
  fes_gpu = [cla(T, queue, fe) for fe in fes]
  scales = [FixedEffects.scale!(zeros(T, fe.n), fe.refs, fe.interaction, sqrtw) for fe in fes]
  caches = [FixedEffects.cache!(zeros(T, length(sqrtw)), fe.refs, fe.interaction, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
  scales = [cla(T, queue, scale) for scale in scales]
  caches = [cla(T, queue, cache) for cache in caches]
  sqrtw = cla(T, queue, sqrtw)
  xs = FixedEffects.FixedEffectCoefficients([clazeros(T, queue, fe.n) for fe in fes_gpu])
  v = FixedEffects.FixedEffectCoefficients([clazeros(T, queue, fe.n) for fe in fes_gpu])
  h = FixedEffects.FixedEffectCoefficients([clazeros(T, queue, fe.n) for fe in fes_gpu])
  hbar = FixedEffects.FixedEffectCoefficients([clazeros(T, queue, fe.n) for fe in fes_gpu])
  u = clazeros(T, queue, length(sqrtw))
  r = clazeros(T, queue, length(sqrtw))
  FixedEffectLSMRCL(FixedEffects.FixedEffectLSMR(fes_gpu, scales, caches, xs, v, h, hbar, u, r, sqrtw), zeros(T, length(sqrtw)), fes)
end





function FixedEffects.solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRCL; kwargs...)
  copyto!(feM.tmp, r)
  copyto!(feM.m.r, feM.tmp)
  iterations, converged = FixedEffects.solve!(feM.m, feM.m.r; kwargs...)
  mul!(feM.m.r, feMm.m, feM.m.xs, -1.0, 1.0)
  copyto!(feM.tmp, feM.m.r)
  copyto!(r, feM.tmp)
  r, iterations, converged
end


using FixedEffects, CategoricalArrays, Random, Statistics
Random.seed!(1234)
N = 10000
K = 100
id1 = categorical(Int.(rand(1:(N/K), N)))
id2 = categorical(Int.(rand(1:K, N)))
id3 = categorical(Int.(rand(1:(N/K), N)))
x = rand(N)
X = [x x x x x x x x x x]

fes = [FixedEffect(id1), FixedEffect(id2)]
sqrtw = ones(N)
feM = AbstractFixedEffectMatrix{Float32}(fes, sqrtw, Val{:lsmr_cl})
solve_residuals!(x, feM)





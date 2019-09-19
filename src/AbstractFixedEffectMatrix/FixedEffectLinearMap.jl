##############################################################################
## 
## FixedEffectCoefficients : vector x in A'Ax = A'b
##
## We define these methods used in lsmr! (duck typing): 
## copyto!, fill!, rmul!, axpy!, norm
##
##############################################################################

abstract type AbstractFixedEffectCoefficients end
  
struct FixedEffectCoefficients{T<:AbstractVector} <: AbstractFixedEffectCoefficients
    x::Vector{T}
end
Base.iterate(xs::FixedEffectCoefficients) = iterate(xs.x)
Base.iterate(xs::FixedEffectCoefficients, state) = iterate(xs.x, state)

function FixedEffectCoefficients(fes::Vector{T}) where {T<:FixedEffect}
    FixedEffectCoefficients([zeros(fe.n) for fe in fes])
end


eltype(xs::FixedEffectCoefficients) = eltype(eltype(xs.x))
length(xs::FixedEffectCoefficients) = sum(length(x) for x in xs)
norm(xs::FixedEffectCoefficients) = sqrt(sum(sum(abs2, x) for x in xs))

function fill!(xs::FixedEffectCoefficients, α::Number)
    for x in xs
        fill!(x, α)
    end
    return xs
end

function rmul!(xs::FixedEffectCoefficients, α::Number)
    for x in xs
        rmul!(x, α)
    end
    return xs
end

function copyto!(xs1::FixedEffectCoefficients, xs2::FixedEffectCoefficients)
    for (x1, x2) in zip(xs1, xs2)
        copyto!(x1, x2)
    end
    return xs1
end

function axpy!(α::Number, xs1::FixedEffectCoefficients, xs2::FixedEffectCoefficients)
    for (x1, x2) in zip(xs1, xs2)
        axpy!(α, x1, x2)
    end
    return xs2
end

##############################################################################
## 
## FixedEffectLSMR
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## mul!
##
##############################################################################

abstract type AbstractFixedEffectLSMR <: AbstractFixedEffectMatrix
end

struct FixedEffectLSMR <: AbstractFixedEffectLSMR
    fes::Vector{<:FixedEffect}
    scales::Vector{AbstractVector}
    caches::Vector{AbstractVector}
    xs::AbstractFixedEffectCoefficients
    v::AbstractFixedEffectCoefficients
    h::AbstractFixedEffectCoefficients
    hbar::AbstractFixedEffectCoefficients
    u::AbstractVector
    sqrtw::AbstractVector
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr}})
    scales = [_scale(fe, sqrtw) for fe in fes] 
    caches = [_cache(fe, scale, sqrtw) for (fe, scale) in zip(fes, scales)]
    x = FixedEffectCoefficients(fes)
    v = FixedEffectCoefficients(fes)
    h = FixedEffectCoefficients(fes)
    hbar = FixedEffectCoefficients(fes)
    #u = similar(first(fes), length(first(fes)))
  #u .= zero(eltype(first(fes)))
    u = zeros(length(first(fes)))
    return FixedEffectLSMR(fes, scales, caches, x, v, h, hbar, u, sqrtw)
end

function _scale(x::FixedEffect, sqrtw)
    out = zeros(x.n)
    for i in eachindex(x.refs)
        out[x.refs[i]] += abs2(x.interaction[i] * sqrtw[i])
    end
    for i in eachindex(out)
        out[i] = out[i] > 0.0 ? (1.0 / sqrt(out[i])) : 0.0
    end
    return out
end

function _cache(fe::FixedEffect, scale, sqrtw::AbstractVector)
    out = zeros(length(fe.refs))
    @inbounds @simd for i in eachindex(out)
        out[i] = scale[fe.refs[i]] * fe.interaction[i] * sqrtw[i]
    end
    return out
end

eltype(fem::FixedEffectLSMR) = eltype(fem.xs)
adjoint(fem::FixedEffectLSMR) = Adjoint(fem)

function size(fem::FixedEffectLSMR, dim::Integer)
    (dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

function mul!(y::AbstractVector, fem::AbstractFixedEffectLSMR, 
              fecoefs::AbstractFixedEffectCoefficients, α::Number, β::Number)
    rmul!(y, β)
    for (fecoef, fe, cache) in zip(fecoefs, fem.fes, fem.caches)
        demean!(y, fecoef, fe.refs, α, cache)
    end
    return y
end

function demean!(y::AbstractVector, fecoef::AbstractVector, refs::AbstractVector, 
                 α::Number, cache::AbstractVector)
  @simd ivdep for i in eachindex(y)
    @inbounds y[i] += fecoef[refs[i]] * α * cache[i]
  end
end

function mul!(fecoefs::AbstractFixedEffectCoefficients,
              Cfem::Adjoint{T, FEM},
              y::AbstractVector, α::Number, β::Number) where{T, FEM<:AbstractFixedEffectLSMR}
    fem = adjoint(Cfem)
    rmul!(fecoefs, β)
    for (fecoef, fe, cache) in zip(fecoefs, fem.fes, fem.caches)
        mean!(fecoef, fe.refs, y, α, cache, fe.n)
    end
    return fecoefs
end

function mean!(fecoef::AbstractVector, refs::AbstractVector, y::AbstractVector, 
               α::Number, cache::AbstractVector, nrefs::Integer)
    @simd ivdep for i in eachindex(y)
        @inbounds fecoef[refs[i]] += y[i] * α * cache[i]
    end
end

function solve!(feM::FixedEffectLSMR, r::AbstractVector; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(feM.xs, 0.0)
    copyto!(feM.u, r)
    x, ch = lsmr!(feM.xs, feM, feM.u, feM.v, feM.h, feM.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

##############################################################################
##
## GPU interface
##
##############################################################################

using CuArrays, CUDAnative, CUDAdrv

CuArray([1])
# Convert containers into GPU                       
function CuArray(fe::FixedEffect)
  FixedEffect{CuArray{eltype(fe.refs),1}, CuArray{eltype(fe.interaction),1}}(CuArray(fe.refs), CuArray(fe.interaction), fe.n)
end

function CuArray(c::AbstractFixedEffectCoefficients)
  FixedEffectCoefficients(CuArray.(c.x))
end

function CuArray(m::AbstractFixedEffectLSMR)
  FixedEffectLSMR(CuArray.(m.fes), CuArray.(m.scales),
                  CuArray.(m.caches), CuArray(m.xs),
                  CuArray(m.v), CuArray(m.h),
                  CuArray(m.hbar),
                  CuArray(m.u), CuArray(m.sqrtw)) 
end


function mean!(fecoef::CuArray{R,1}, refs::CuArray{I,1},
               y::CuArray{R,1}, α::Number,
               cache::CuArray{R,1}, nrefs::Integer) where{R<:Real, I<:Integer}  
  nthreads = 256
  nblocks = ceil(Int, length(y)/nthreads)
  #tmp = CuArray(zeros(R,length(y), nrefs))
  #println("calling cuda mean")
  @cuda threads=nthreads blocks=nblocks meankernel!(fecoef,refs,y,α, cache)
  #fecoef .+= @view sum(tmp,dims=1)[:]
end
function meankernel!(fecoef, refs, y, α, cache)
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = blockDim().x * gridDim().x
  @inbounds for i = index:stride:length(y)
    r = refs[i]
    CUDAnative.atomic_add!(pointer(fecoef, r), y[i]*α*cache[i])
  end
  return nothing
end

function demean!(y::CuArray{R,1}, fecoef::CuArray{R,1},
                 refs::CuArray{I,1}, 
                 α::Number, cache::CuArray{R,1}) where{R<:Real, I<:Integer}
  nthreads = 256
  nblocks = ceil(Int, length(y)/nthreads)
  #println("calling cuda demean")
  @cuda threads=nthreads blocks=nblocks demeankernel!(fecoef,refs,y,α, cache)
end
function demeankernel!(fecoef, refs, y, α, cache)
  index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = blockDim().x * gridDim().x
  for i = index:stride:length(y)
    @inbounds y[i] += fecoef[refs[i]]*α*cache[i]
  end
  return nothing
end


@inline function reduce_warp(op::F, val::T)::T where {F<:Function,T}
  offset = CUDAnative.warpsize() ÷ 2
  while offset > 0
    val = op(val, shfl_down(val, offset))
    offset ÷= 2
  end
  return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::F, val::T)::T where {F<:Function,T}
  # shared mem for 32 partial sums
  shared = @cuStaticSharedMem(T, 32)
  
  wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())

  # each warp performs partial reduction
  val = reduce_warp(op, val)

  # write reduced value to shared memory
  if lane == 1
    @inbounds shared[wid] = val
  end

  # wait for all partial reductions
  sync_threads()

  # read from shared memory only if that warp existed
  @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)

  # final reduce within first warp
  if wid == 1
    val = reduce_warp(op, val)
  end

  return val
end

# Reduce an array across a complete grid
function reduce_grid(op::F, input::CuDeviceVector{T}, output::CuDeviceVector{T},
                     len::Integer) where {F<:Function,T}
  # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
  val = zero(T)

  # reduce multiple elements per thread (grid-stride loop)
  # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  step = blockDim().x * gridDim().x
  while i <= len
    @inbounds val = op(val, input[i])
    i += step
  end

  val = reduce_block(op, val)

  if threadIdx().x == 1
    @inbounds output[blockIdx().x] = val
  end

  return
end



##############################################################################
##
## Implement AbstractFixedEffectMatrix interface
##
##############################################################################\

function solve_residuals!(r::AbstractVector{Float64}, feM::FixedEffectLSMR; kwargs...)
    r .*= feM.sqrtw
    iterations, converged = solve!(feM, r; kwargs...)
    mul!(r, feM, feM.xs, -1.0, 1.0)
    r ./=  feM.sqrtw
    return r, iterations, converged
end

function solve_coefficients!(r::AbstractVector{Float64}, feM::FixedEffectLSMR; kwargs...)
    r .*= feM.sqrtw
    iterations, converged = solve!(feM, r; kwargs...)
    for (x, scale) in zip(feM.xs, feM.scales)
        x .*=  scale
    end 
    full(normalize!(feM.xs.x, feM.fes; kwargs...), feM.fes), iterations, converged
end

##############################################################################
##
## LSMR Parallel
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct FixedEffectLSMRParallel{W} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::W
end

function FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, 
                ::Type{Val{:lsmr_parallel}})
    FixedEffectLSMRParallel(fes, sqrtw)
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRParallel; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectLSMRParallel; kwargs...)
    iterations = zeros(Int, size(X, 2))
    convergeds = zeros(Bool, size(X, 2))
    result = pmap(x -> solve_residuals!(x, feM; kwargs...), [X[:, j] for j in 1:size(X, 2)])
    for j in 1:size(X, 2)
        X[:, j] = result[j][1]
        iterations[j] =  result[j][2]
        convergeds[j] = result[j][3]
    end
    return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRParallel, ; kwargs...)
    solve_coefficients!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct FixedEffectLSMRThreads{W} <: AbstractFixedEffectMatrix
    fes::Vector{<:FixedEffect}
    sqrtw::W
end

FixedEffectMatrix(fes::Vector{<:FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:lsmr_threads}}) = FixedEffectLSMRThreads(fes, sqrtw)

function solve_residuals!(r::AbstractVector, feM::FixedEffectLSMRThreads; kwargs...)
    solve_residuals!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end

function solve_residuals!(X::AbstractMatrix, feM::FixedEffectLSMRThreads; kwargs...)
   iterations = zeros(Int, size(X, 2))
   convergeds = zeros(Bool, size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        X[:, j], iteration, converged = solve_residuals!(X[:, j], feM; kwargs...)
        iterations[j] = iteration
        convergeds[j] = converged
   end
   return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectLSMRThreads; kwargs...)
    solve_coefficients!(r, FixedEffectMatrix(feM.fes, feM.sqrtw, Val{:lsmr}); kwargs...)
end


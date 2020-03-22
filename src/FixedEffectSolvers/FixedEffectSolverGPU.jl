using CuArrays.CUDAnative
import CuArrays: allowscalar
allowscalar(false)

##############################################################################
##
## Conversion FixedEffect between CPU and GPU
##
##############################################################################

# https://github.com/JuliaGPU/CuArrays.jl/issues/306
cuzeros(T::Type, n::Integer) = fill!(CuVector{T}(undef, n), zero(T))
function CuArrays.cu(T::Type, fe::FixedEffect)
	refs = CuArray(fe.refs)
	interaction = cu(T, fe.interaction)
	FixedEffect{typeof(refs), typeof(interaction)}(refs, interaction, fe.n)
end
CuArrays.cu(T::Type, w::Union{Fill, Ones, Zeros}) = fill!(CuVector{T}(undef, length(w)), w[1])
CuArrays.cu(T::Type, w::AbstractVector) = CuVector{T}(convert(Vector{T}, w))

##############################################################################
##
## FixedEffectLinearMap on the GPU (code by Paul Schrimpf)
##
## Model matrix of categorical variables
## mutiplied by diag(1/sqrt(∑w * interaction^2, ..., ∑w * interaction^2) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## eltyp
## size
## mul!
##
##############################################################################

mutable struct FixedEffectLinearMapGPU{T}
	fes::Vector{<:FixedEffect}
	scales::Vector{<:AbstractVector}
	caches::Vector{<:AbstractVector}
	nthreads::Int
end

function FixedEffectLinearMapGPU{T}(fes::Vector{<:FixedEffect}, ::Type{Val{:gpu}}) where {T}
	fes = [cu(T, fe) for fe in fes]
	scales = [cuzeros(T, fe.n) for fe in fes]
	caches = [cuzeros(T, length(fes[1].interaction)) for fe in fes]
	return FixedEffectLinearMapGPU{T}(fes, scales, caches, 256)
end

function scale!(scale::CuVector, refs::CuVector, interaction::CuVector, sqrtw::CuVector, nthreads::Integer)
	nblocks = cld(length(refs), nthreads) 
	@cuda threads=nthreads blocks=nblocks scale!_kernel!(scale, refs, interaction, sqrtw)
	scale .= sqrt.(scale)
end

function scale!_kernel!(scale, refs, interaction, sqrtw)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(interaction)
		CuArrays.CUDAnative.atomic_add!(pointer(scale, refs[i]), abs2(interaction[i] * sqrtw[i]))
	end
end

function cache!(cache::CuVector, refs::CuVector, interaction::CuVector, sqrtw::CuVector, scale::CuVector, nthreads::Integer)
	nblocks = cld(length(cache), nthreads) 
	@cuda threads=nthreads blocks=nblocks cache!_kernel!(cache, refs, interaction, sqrtw, scale)
end

function cache!_kernel!(cache, refs, interaction, sqrtw, scale)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(cache)
		cache[i] += interaction[i] * sqrtw[i] / scale[refs[i]]
	end
end

LinearAlgebra.adjoint(fem::FixedEffectLinearMapGPU) = Adjoint(fem)

function Base.size(fem::FixedEffectLinearMapGPU, dim::Integer)
	(dim == 1) ? length(fem.fes[1].refs) : (dim == 2) ? sum(fe.n for fe in fem.fes) : 1
end

Base.eltype(x::FixedEffectLinearMapGPU{T}) where {T} = T

function LinearAlgebra.mul!(fecoefs::FixedEffectCoefficients, 
	Cfem::Adjoint{T, FixedEffectLinearMapGPU{T}},
	y::AbstractVector, α::Number, β::Number) where {T}
	fem = adjoint(Cfem)
	rmul!(fecoefs, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		gather!(fecoef, fe.refs, α, y, cache, fem.nthreads)
	end
	return fecoefs
end


function gather!(fecoef::CuVector, refs::CuVector, α::Number, y::CuVector, cache::CuVector, nthreads::Integer)
	nblocks = cld(length(y), nthreads) 
	@cuda threads=nthreads blocks=nblocks gather_kernel!(fecoef, refs, α, y, cache)    
end

function gather_kernel!(fecoef, refs, α, y, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		CuArrays.CUDAnative.atomic_add!(pointer(fecoef, refs[i]), α * y[i] * cache[i])
	end
end

function LinearAlgebra.mul!(y::AbstractVector, fem::FixedEffectLinearMapGPU, 
			  fecoefs::FixedEffectCoefficients, α::Number, β::Number)
	rmul!(y, β)
	for (fecoef, fe, cache) in zip(fecoefs.x, fem.fes, fem.caches)
		scatter!(y, α, fecoef, fe.refs, cache, fem.nthreads)
	end
	return y
end

function scatter!(y::CuVector, α::Number, fecoef::CuVector, refs::CuVector, cache::CuVector, nthreads::Integer)
	nblocks = cld(length(y), nthreads)
	@cuda threads=nthreads blocks=nblocks scatter_kernel!(y, α, fecoef, refs, cache)
end

function scatter_kernel!(y, α, fecoef, refs, cache)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = blockDim().x * gridDim().x
	@inbounds for i = index:stride:length(y)
		y[i] += α * fecoef[refs[i]] * cache[i]
	end
end

##############################################################################
##
## Implement AbstractFixedEffectSolver interface
##
##############################################################################

mutable struct FixedEffectSolverGPU{T} <: AbstractFixedEffectSolver{T}
	m::FixedEffectLinearMapGPU{T}
	sqrtw::CuVector{T}
	b::CuVector{T}
	r::CuVector{T}
	x::FixedEffectCoefficients{T}
	v::FixedEffectCoefficients{T}
	h::FixedEffectCoefficients{T}
	hbar::FixedEffectCoefficients{T}
	tmp::Vector{T} # used to convert AbstractVector to Vector{T}
	fes::Vector{<:FixedEffect}
end
	
function AbstractFixedEffectSolver{T}(fes::Vector{<:FixedEffect}, weights::AbstractWeights, ::Type{Val{:gpu}}) where {T}
	m = FixedEffectLinearMapGPU{T}(fes, Val{:gpu})
	b = cuzeros(T, length(weights))
	r = cuzeros(T, length(weights))
	x = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	v = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	h = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	hbar = FixedEffectCoefficients([cuzeros(T, fe.n) for fe in fes])
	tmp = zeros(T, length(weights))
	update_weights!(FixedEffectSolverGPU{T}(m, cuzeros(T, length(weights)), b, r, x, v, h, hbar, tmp, fes), weights)
end


function update_weights!(feM::FixedEffectSolverGPU{T}, weights::AbstractWeights) where {T}
	sqrtw = cu(T, sqrt.(weights))
	for (scale, fe) in zip(feM.m.scales, feM.m.fes)
		scale!(scale, fe.refs, fe.interaction, sqrtw, 256)
	end
	for (cache, scale, fe) in zip(feM.m.caches, feM.m.scales, feM.m.fes)
		cache!(cache, fe.refs, fe.interaction, sqrtw, scale, 256)
	end	
	feM.sqrtw = sqrtw
	return feM
end

function solve_residuals!(r::AbstractVector, feM::FixedEffectSolverGPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.tmp, r)
	copyto!(feM.r, feM.tmp)
	feM.r .*= feM.sqrtw
	copyto!(feM.b, feM.r)
	fill!(feM.x, 0.0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	mul!(feM.r, feM.m, feM.x, -1.0, 1.0)
	feM.r ./=  feM.sqrtw
	copyto!(feM.tmp, feM.r)
	copyto!(r, feM.tmp)
	return r, div(ch.mvps, 2), ch.isconverged
end

function FixedEffects.solve_residuals!(X::AbstractMatrix, feM::FixedEffects.FixedEffectSolverGPU; kwargs...)
    iterations = Int[]
    convergeds = Bool[]
    for j in 1:size(X, 2)
        _, iteration, converged = solve_residuals!(view(X, :, j), feM; kwargs...)
        push!(iterations, iteration)
        push!(convergeds, converged)
    end
    return X, iterations, convergeds
end

function solve_coefficients!(r::AbstractVector, feM::FixedEffectSolverGPU{T}; tol::Real = sqrt(eps(T)), maxiter::Integer = 100_000) where {T}
	copyto!(feM.tmp, r)
	copyto!(feM.b, feM.tmp)
	feM.b .*= feM.sqrtw
	fill!(feM.x, 0.0)
	x, ch = lsmr!(feM.x, feM.m, feM.b, feM.v, feM.h, feM.hbar; atol = tol, btol = tol, maxiter = maxiter)
	for (x, scale) in zip(feM.x.x, feM.m.scales)
		x ./=  scale
	end
	x = Vector{eltype(r)}[collect(x) for x in feM.x.x]
	full(normalize!(x, feM.fes; tol = tol, maxiter = maxiter), feM.fes), div(ch.mvps, 2), ch.isconverged
end
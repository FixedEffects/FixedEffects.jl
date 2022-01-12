function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    
    Base.precompile(Tuple{Core.kwftype(typeof(FixedEffects.normalize!)),NamedTuple{(:tol, :maxiter),Tuple{Float64,Int64}},typeof(FixedEffects.normalize!),Array{Array{Float64,1},1},Array{FixedEffects.FixedEffect,1}})
    Base.precompile(Tuple{Core.kwftype(typeof(FixedEffects.normalize!)),NamedTuple{(:tol, :maxiter),Tuple{Float64,Int64}},typeof(FixedEffects.normalize!),Array{Array{Float64,1},1},Array{FixedEffect{Array{UInt32,1},UnitWeights{Float64}},1}})
    Base.precompile(Tuple{typeof(FixedEffects.full),Array{Array{Float64,1},1},Array{FixedEffects.FixedEffect,1}})

    Base.precompile(Tuple{Type{FixedEffects.AbstractFixedEffectSolver{Float64}},Array{FixedEffects.FixedEffect,1},Weights{Float64,Float64,Array{Float64,1}},Type{Val{:cpu}}})
    Base.precompile(Tuple{Type{FixedEffects.AbstractFixedEffectSolver{Float64}},Array{FixedEffects.FixedEffect,1},UnitWeights{Float64},Type{Val{:cpu}}})
    Base.precompile(Tuple{Type{FixedEffects.AbstractFixedEffectSolver{Float32}},Array{FixedEffects.FixedEffect,1},Weights{Float64,Float64,Array{Float64,1}},Type{Val{:cpu}},Int64})
    Base.precompile(Tuple{typeof(FixedEffects.update_weights!),FixedEffects.FixedEffectSolverCPU{Float64},Weights{Int64,Int64,Array{Int64,1}}})


    Base.precompile(Tuple{Core.kwftype(typeof(FixedEffects.solve_coefficients!)),NamedTuple{(:tol, :maxiter),Tuple{Float64,Int64}},typeof(FixedEffects.solve_coefficients!),Array{Float64,1},FixedEffects.FixedEffectSolverCPU{Float64}})

    Base.precompile(Tuple{typeof(solve_residuals!),Array{Float64,1},Array{FixedEffect{Array{UInt32,1},UnitWeights{Float64}},1}})
    Base.precompile(Tuple{typeof(solve_residuals!),Array{Float64,2},Array{FixedEffect{Array{UInt32,1},UnitWeights{Float64}},1}})
    Base.precompile(Tuple{Core.kwftype(typeof(FixedEffects.solve_residuals!)),NamedTuple{(:maxiter, :tol),Tuple{Int64,Float64}},typeof(solve_residuals!),Array{Float64,1},FixedEffects.FixedEffectSolverCPU{Float64}})
    
    
    Base.precompile(Tuple{Core.kwftype(typeof(FixedEffects.lsmr!)),NamedTuple{(:atol, :btol, :maxiter),Tuple{Float64,Float64,Int64}},typeof(FixedEffects.lsmr!),FixedEffects.FixedEffectCoefficients{Array{Float64,1}},FixedEffects.FixedEffectLinearMapCPU{Float64},Array{Float64,1},FixedEffects.FixedEffectCoefficients{Array{Float64,1}},FixedEffects.FixedEffectCoefficients{Array{Float64,1}},FixedEffects.FixedEffectCoefficients{Array{Float64,1}}})
    Base.precompile(Tuple{Core.kwftype(typeof(FixedEffects.lsmr!)),NamedTuple{(:atol, :btol, :maxiter),Tuple{Float64,Float64,Int64}},typeof(FixedEffects.lsmr!),FixedEffects.FixedEffectCoefficients{Array{Float32,1}},FixedEffects.FixedEffectLinearMapCPU{Float32},Array{Float32,1},FixedEffects.FixedEffectCoefficients{Array{Float32,1}},FixedEffects.FixedEffectCoefficients{Array{Float32,1}},FixedEffects.FixedEffectCoefficients{Array{Float32,1}}}) 
end

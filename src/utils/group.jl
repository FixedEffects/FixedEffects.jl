
##############################################################################
##
## group transform multiple CategoricalVector into one
## Output is a CategoricalVector where pool is type Int64, equal to ranking of group
## Missing in some row mean result has Missing on this row
## 
##############################################################################

function group(args...)
	v = categorical(args[1])
	if length(args) == 1
		x = v.refs
	else
		x, ngroups = convert(Vector{UInt}, v.refs), length(levels(v))
		for j = 2:length(args)
			v = categorical(args[j])
			x, ngroups = pool_combine!(x, v, ngroups)
		end
	end
	factorize!(x)
end

#  drop unused levels
function factorize!(refs::Vector{T}) where {T}
	uu = unique(refs)
	sort!(uu)
	has_missing = uu[1] == 0
	dict = Dict{T, Int}(zip(uu, (1-has_missing):(length(uu)-has_missing)))
	newrefs = zeros(UInt32, length(refs))
	for i in 1:length(refs)
		 newrefs[i] = dict[refs[i]]
	end
	if has_missing
		Tout = Union{Int, Missing}
	else
		Tout = Int
	end
	CategoricalArray{Tout, 1}(newrefs, CategoricalPool(collect(1:(length(uu)-has_missing))))
end
function pool_combine!(x::Array{T, N}, dv::CategoricalVector, ngroups::Integer) where {T, N}
	for i in 1:length(x)
	    # if previous one is NA or this one is NA, set to NA
	    x[i] = (dv.refs[i] == 0 || x[i] == zero(T)) ? zero(T) : x[i] + (dv.refs[i] - 1) * ngroups
	end
	return x, ngroups * length(levels(dv))
end
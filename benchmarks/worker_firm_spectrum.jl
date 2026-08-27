# Measure the leading canonical correlations sigma_k between the two FE subspaces for the
# benchmark worker-firm data. After Jacobi scaling A'A = [[I,C],[C',I]] with sigma_k = svd(C);
# slow LSMR convergence is driven by sigma_k -> 1. This quantifies how many slow modes exist
# (=> how many deflation vectors k would be needed and the achievable iteration reduction).
using Random, LinearAlgebra
Random.seed!(1234)

# --- benchmark hard scenario ---
N = 800_000; M = 400_000; O = 50_000
refs1 = rand(1:M, N)                                             # worker
refs2 = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in refs1]  # firm
n1 = maximum(refs1); n2 = maximum(refs2)

# Jacobi scales (unweighted, no interaction): scale[g] = 1/sqrt(group count)
function scales(refs, n)
    c = zeros(Int, n); @inbounds for r in refs; c[r] += 1; end
    s = zeros(n); @inbounds for g in 1:n; s[g] = c[g] > 0 ? 1 / sqrt(c[g]) : 0.0; end
    s
end
s1 = scales(refs1, n1); s2 = scales(refs2, n2)

# C v = gather_FE1(scatter_FE2(v)), maps firm-space (n2) -> worker-space (n1)
function Cmul!(out, v, refs1, refs2, s1, s2)
    fill!(out, 0.0)
    @inbounds for i in eachindex(refs1)
        out[refs1[i]] += s2[refs2[i]] * v[refs2[i]]
    end
    @inbounds for g in eachindex(out); out[g] *= s1[g]; end
    out
end

# C' u, maps worker-space (n1) -> firm-space (n2)
function Ctmul!(out, u, refs1, refs2, s1, s2)
    fill!(out, 0.0)
    @inbounds for i in eachindex(refs1)
        out[refs2[i]] += s1[refs1[i]] * u[refs1[i]]
    end
    @inbounds for h in eachindex(out); out[h] *= s2[h]; end
    out
end

# Subspace (block) iteration on C'C (acts on firm space n2=50k) for the top-k sigma^2.
function top_sigmas(k, iters)
    Y = randn(n2, k)
    tmp1 = zeros(n1); ritz = zeros(k)
    for it in 1:iters
        # Z = (C'C) Y
        Z = similar(Y)
        for j in 1:k
            Cmul!(tmp1, view(Y, :, j), refs1, refs2, s1, s2)
            Ctmul!(view(Z, :, j), tmp1, refs1, refs2, s1, s2)
        end
        F = qr(Z); Q = Matrix(F.Q)
        # Rayleigh-Ritz on Q
        AQ = similar(Q)
        for j in 1:k
            Cmul!(tmp1, view(Q, :, j), refs1, refs2, s1, s2)
            Ctmul!(view(AQ, :, j), tmp1, refs1, refs2, s1, s2)
        end
        H = Symmetric(Q' * AQ)
        E = eigen(H); ev = E.values; perm = sortperm(ev, rev = true)
        ritz = ev[perm]
        Y = Q * E.vectors[:, perm]
    end
    sqrt.(clamp.(ritz, 0, Inf))
end

println("worker-firm: N=$N, n1(worker)=$n1, n2(firm)=$n2")
k = 30
sig = top_sigmas(k, 60)
println("\nTop $k canonical correlations sigma_k (descending):")
for (j, s) in enumerate(sig)
    gap = 1 - s
    println("  k=$(lpad(j, 2))  sigma=", round(s, digits = 6), "   1-sigma=", round(gap, sigdigits = 3))
end

# condition-number proxy and crude iteration estimate (iters ~ sqrt((1+s)/(1-s)) for sigma_max)
s2nd = sig[2]
println("\nsigma_1=", round(sig[1], digits = 6), " (constant mode, deflated by rank-deficiency)")
println("sigma_2=", round(s2nd, digits = 6), "  => kappa~", round((1 + s2nd) / (1 - s2nd), digits = 1),
        "  sqrt(kappa)~", round(sqrt((1 + s2nd) / (1 - s2nd)), digits = 1))
nbig = count(>(0.99), sig)
println("count(sigma>0.99) in top $k: ", nbig)
nbig999 = count(>(0.999), sig)
println("count(sigma>0.999) in top $k: ", nbig999)

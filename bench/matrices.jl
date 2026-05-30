# Structured test-matrix generators (pure Julia, no external data needed).
using SparseArrays, LinearAlgebra, Random

"5-point Laplacian on a k×k grid, n = k^2 (SPD, classic sparse-LU benchmark)."
function poisson2d(k::Int)
    n = k * k
    I = Int[]
    J = Int[]
    V = Float64[]
    idx(i, j) = (j - 1) * k + i
    for j in 1:k, i in 1:k
        c = idx(i, j)
        push!(I, c)
        push!(J, c)
        push!(V, 4.0)
        i > 1 && (push!(I, c); push!(J, idx(i - 1, j)); push!(V, -1.0))
        i < k && (push!(I, c); push!(J, idx(i + 1, j)); push!(V, -1.0))
        j > 1 && (push!(I, c); push!(J, idx(i, j - 1)); push!(V, -1.0))
        j < k && (push!(I, c); push!(J, idx(i, j + 1)); push!(V, -1.0))
    end
    return sparse(I, J, V, n, n)
end

"7-point Laplacian on a k×k×k grid, n = k^3."
function poisson3d(k::Int)
    n = k^3
    I = Int[]
    J = Int[]
    V = Float64[]
    idx(i, j, l) = ((l - 1) * k + (j - 1)) * k + i
    for l in 1:k, j in 1:k, i in 1:k
        c = idx(i, j, l)
        push!(I, c)
        push!(J, c)
        push!(V, 6.0)
        i > 1 && (push!(I, c); push!(J, idx(i - 1, j, l)); push!(V, -1.0))
        i < k && (push!(I, c); push!(J, idx(i + 1, j, l)); push!(V, -1.0))
        j > 1 && (push!(I, c); push!(J, idx(i, j - 1, l)); push!(V, -1.0))
        j < k && (push!(I, c); push!(J, idx(i, j + 1, l)); push!(V, -1.0))
        l > 1 && (push!(I, c); push!(J, idx(i, j, l - 1)); push!(V, -1.0))
        l < k && (push!(I, c); push!(J, idx(i, j, l + 1)); push!(V, -1.0))
    end
    return sparse(I, J, V, n, n)
end

"Random unsymmetric sparse matrix with ~`deg` nonzeros/column + strong diagonal."
function randmat(n::Int, deg::Int; seed = 1)
    rng = MersenneTwister(seed)
    A = sprand(rng, n, n, deg / n)
    return A + sparse((deg + 1.0) * I, n, n)
end

"Unsymmetric matrix with a moderately fill-prone structure (arrowhead + band)."
function arrowband(n::Int, bw::Int; seed = 3)
    rng = MersenneTwister(seed)
    I = Int[]
    J = Int[]
    V = Float64[]
    for j in 1:n
        push!(I, j)
        push!(J, j)
        push!(V, n + 0.0)
        for d in 1:bw
            if j + d <= n
                push!(I, j)
                push!(J, j + d)
                push!(V, randn(rng))
                push!(I, j + d)
                push!(J, j)
                push!(V, randn(rng))
            end
        end
    end
    # sparse arrowhead row/col
    for j in 1:n
        if rand(rng) < 0.3
            push!(I, 1)
            push!(J, j)
            push!(V, randn(rng))
            push!(I, j)
            push!(J, 1)
            push!(V, randn(rng))
        end
    end
    return sparse(I, J, V, n, n)
end

function testset()
    mats = Tuple{String, SparseMatrixCSC{Float64, Int}}[]
    push!(mats, ("poisson2d-32 (n=1024)", poisson2d(32)))
    push!(mats, ("poisson2d-64 (n=4096)", poisson2d(64)))
    push!(mats, ("poisson2d-100(n=10000)", poisson2d(100)))
    push!(mats, ("poisson3d-16 (n=4096)", poisson3d(16)))
    push!(mats, ("poisson3d-24 (n=13824)", poisson3d(24)))
    push!(mats, ("rand n=2000 deg8", randmat(2000, 8)))
    push!(mats, ("rand n=10000 deg10", randmat(10000, 10)))
    push!(mats, ("arrowband n=4000 bw4", arrowband(4000, 4)))
    return mats
end

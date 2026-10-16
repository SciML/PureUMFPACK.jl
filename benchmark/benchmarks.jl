using PureUMFPACK, BenchmarkTools
using StableRNGs, SparseArrays, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

n = 1000
A = sprand(rng, n, n, 0.01) + 2.0 * sparse(I, n, n)
b = rand(rng, n)
x = similar(b)

# =============================================================================
# LU factorization (GPLU + multifrontal)
# =============================================================================

SUITE["factorize"] = BenchmarkGroup()

SUITE["factorize"]["gplu"] = @benchmarkable gplu($A)
SUITE["factorize"]["splu"] = @benchmarkable splu($A)
SUITE["factorize"]["multifrontal_lu"] = @benchmarkable multifrontal_lu($A)
SUITE["factorize"]["gplu_tol0"] = @benchmarkable gplu($A; tol = 0.0)

# =============================================================================
# Solve
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

F = gplu(A)
SUITE["solve"]["ldiv"] = @benchmarkable $F \ $b

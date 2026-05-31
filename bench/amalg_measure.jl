# Relaxed supernode amalgamation sweep: for each matrix class and each `relax`
# level, report supernode count, total fill nnz(L)+nnz(U), and best-of numeric
# factorization time.  `relax = 0` reproduces the fundamental partition exactly.
#
#   julia --project=. bench/amalg_measure.jl

using PureUMFPACK
using PureUMFPACK: symbolic_mf, predicted_fill, multifrontal_lu
using SparseArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "matrices.jl"))

bestof(f, reps) = minimum(@elapsed(f()) for _ in 1:reps)

function measure(name, A, relax)
    S = symbolic_mf(A; relax = relax)
    nsuper = length(S.sstart) - 1
    F = multifrontal_lu(A; relax = relax)
    nlu = nnz(F.L) + nnz(F.U)
    @assert predicted_fill(S) == nlu "predicted_fill mismatch ($name, relax=$relax)"
    t = bestof(() -> multifrontal_lu(A; relax = relax), 4)
    println(rpad(name, 18), " relax=", rpad(relax, 3), " nsuper=", rpad(nsuper, 8),
        " fill=", rpad(nlu, 12), " t=", round(t * 1000, digits = 2), "ms")
    return nothing
end

cases = [
    ("poisson2d(100)", poisson2d(100)),
    ("poisson3d(24)", poisson3d(24)),
    ("randmat(5000,10)", randmat(5000, 10)),
    ("arrowband(5000,5)", arrowband(5000, 5))
]

for (name, A) in cases
    for relax in (0, 1, 2, 4, 8, 16)
        measure(name, A, relax)
    end
    println()
end

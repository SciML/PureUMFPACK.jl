# Attribute the multifrontal allocations: how much is the per-front zeros(nf,nf),
# how much the per-child contribution-block Matrix copy, and what is the steady GC
# fraction (averaged over many reps, not a single noisy @timed).
using PureUMFPACK
using PureUMFPACK: symbolic_mf, multifrontal_lu
using SparseArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "matrices.jl"))
BLAS.set_num_threads(1)

for k in (24, 32, 36)
    A = poisson3d(k)
    n = size(A, 1)
    S = symbolic_mf(A)
    sstart = S.sstart
    colstruct = S.colstruct
    childsupers = S.childsupers
    nsuper = length(sstart) - 1
    # theoretical front + CB bytes from the symbolic structure (8 bytes/Float64)
    frontbytes = 0.0
    cbbytes = 0.0
    for sk in 1:nsuper
        c1 = sstart[sk]
        c2 = sstart[sk + 1] - 1
        np = c2 - c1 + 1
        nu = length(colstruct[c2])
        nf = np + nu
        frontbytes += 8.0 * nf * nf           # F = zeros(Tv,nf,nf) per supernode
        nu > 0 && (cbbytes += 8.0 * nu * nu)    # cb[sk] = Matrix(view(F22)) per supernode
    end
    multifrontal_lu(A; check = false)       # warmup
    tot_alloc = @allocated multifrontal_lu(A; check = false)
    # steady GC fraction over many reps
    reps = n > 30000 ? 8 : 20
    GC.gc()
    t0 = time_ns()
    g0 = Base.gc_time_ns()
    for _ in 1:reps
        multifrontal_lu(A; check = false)
    end
    wall = (time_ns() - t0) / 1e9
    gc = (Base.gc_time_ns() - g0) / 1e9
    @printf("k=%-2d n=%-6d  total_alloc=%.0fMiB  front=%.0fMiB(%.0f%%)  cb=%.0fMiB(%.0f%%)  | over %d reps: wall=%.3gs gc=%.3gs gc_frac=%.1f%%\n",
        k, n, tot_alloc/2^20, frontbytes/2^20, 100 * frontbytes/tot_alloc,
        cbbytes/2^20, 100 * cbbytes/tot_alloc, reps, wall, gc, 100 * gc/wall)
    flush(stdout)
end
println("ALLOC_ATTRIB_DONE")

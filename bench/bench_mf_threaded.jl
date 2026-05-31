# Elimination-tree (subtree) parallelism in the multifrontal kernel: serial vs
# threaded `multifrontal_lu`, and both vs UMFPACK (`lu`).  Interleaved, min-of-N,
# fixed RNG.  BLAS is pinned to one thread so the comparison measures TREE
# parallelism, not nested BLAS.  Run under several Julia threads, e.g.:
#
#   julia -t 8 --project=. bench/bench_mf_threaded.jl
#
# The threaded factor is bit-for-bit identical to the serial one (same fronts,
# same BLAS calls, same extend-add order); `ident` confirms it per case.
using PureUMFPACK
using PureUMFPACK: multifrontal_lu
using SparseArrays, LinearAlgebra, Random
include(joinpath(@__DIR__, "matrices.jl"))

BLAS.set_num_threads(1)

const REPS = parse(Int, get(ENV, "REPS", "15"))

function bestfactor(f, reps)
    f()                                   # warmup / compile
    best = Inf
    for _ in 1:reps
        GC.gc()
        t = @elapsed f()
        t < best && (best = t)
    end
    return best
end

function main()
    sizes = (30, 36)
    println("multifrontal tree parallelism  (REPS=$REPS, 1 BLAS thread, ",
        Threads.nthreads(), " Julia threads)")
    println(rpad("k", 5), rpad("n", 9), rpad("serial(s)", 12), rpad("threaded(s)", 13),
        rpad("umf(s)", 12), rpad("ser/thr", 9), rpad("thr/umf", 9), rpad("ident", 7), "residual")
    for k in sizes
        A = poisson3d(k)
        ser = bestfactor(() -> multifrontal_lu(A; threaded = false), REPS)
        thr = bestfactor(() -> multifrontal_lu(A; threaded = true), REPS)
        umf = bestfactor(() -> lu(A), REPS)
        Fs = multifrontal_lu(A; threaded = false)
        Ft = multifrontal_lu(A; threaded = true)
        ident = Fs.L == Ft.L && Fs.U == Ft.U && Fs.p == Ft.p && Fs.q == Ft.q
        n = size(A, 1)
        b = randn(n)
        x = solve(PureLU(Ft.L, Ft.U, Ft.p, Ft.q, ones(n), A), b)
        res = norm(A * x - b) / norm(b)
        println(rpad(k, 5), rpad(n, 9), rpad(round(ser, digits = 4), 12),
            rpad(round(thr, digits = 4), 13), rpad(round(umf, digits = 4), 12),
            rpad(round(ser / thr, digits = 3), 9), rpad(round(thr / umf, digits = 3), 9),
            rpad(ident, 7), round(res, sigdigits = 3))
    end
end

main()

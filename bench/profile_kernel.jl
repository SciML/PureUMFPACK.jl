using PureUMFPACK
using SparseArrays, LinearAlgebra, Printf, Profile
include(joinpath(@__DIR__, "matrices.jl"))

BLAS.set_num_threads(1)

mats = [("poisson2d-100(n=10000)", poisson2d(100)),
    ("poisson3d-20 (n=8000)", poisson3d(20)),
    ("rand n=5000 deg10", randmat(5000, 10))]

for (name, A) in mats
    F = gplu(A; tol = 0.1)                 # warmup + correctness
    b = randn(size(A, 1))
    res = norm(A * solve(F, b) - b) / norm(b)
    a = @allocated gplu(A; tol = 0.1)
    t = @elapsed (for _ in 1:3
        gplu(A; tol = 0.1)
    end)
    t /= 3
    @printf("%-26s n=%-6d fill=%-9d  fact=%.4gs  alloc=%.1f MiB  res=%.1e\n",
        name, size(A, 1), nnz(F.L)+nnz(F.U), t, a/2^20, res)
    flush(stdout)
end

# Detailed profile on the random matrix (heaviest relative kernel work)
A = randmat(5000, 10)
gplu(A; tol = 0.1)
Profile.clear()
Profile.init(n = 10^7, delay = 0.0005)
@profile (for _ in 1:8
    gplu(A; tol = 0.1)
end)
open(joinpath(@__DIR__, "..", "results", "profile_flat.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 200));
        format = :flat, sortedby = :count, mincount = 15)
end
println("PROFILE WRITTEN")

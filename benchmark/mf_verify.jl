using PureUMFPACK, SparseArrays, LinearAlgebra
include(joinpath(@__DIR__, "matrices.jl"))
ok = true
for (nm, A) in (
        ("p2d-25", poisson2d(25)), ("p3d-6", poisson3d(6)),
        ("p3d-12", poisson3d(12)), ("rand-500", randmat(500, 6)),
        ("arrow-400", arrowband(400, 4)),
    )
    n = size(A, 1)
    b = randn(n)
    xu = lu(A) \ b
    F = multifrontal_lu(A; check = false)
    recon = norm(Matrix(A[F.p, F.q]) - Matrix(F.L * F.U)) / max(1, norm(Matrix(A)))
    x = F \ b
    res = norm(A * x - b) / norm(b)
    mu = norm(x - xu) / norm(xu)
    Lok = istril(F.L) && all(==(1.0), diag(F.L))
    Uok = istriu(F.U)
    bad = recon > 1.0e-10 || res > 1.0e-9 || mu > 1.0e-8 || !Lok || !Uok
    bad && (global ok = false)
    println(
        rpad(nm, 10), " recon=", round(recon, sigdigits = 2),
        " res=", round(res, sigdigits = 2),
        " umf=", round(mu, sigdigits = 2), " Lok=", Lok, " Uok=", Uok, bad ? "  <<BAD" : ""
    )
end
# ComplexF64 (Hermitian-pattern) + Float32 through multifrontal
Ac = sprand(ComplexF64, 300, 300, 8 / 300) + (5 + 0im) * I
Ac = Ac + Ac'
bc = randn(ComplexF64, 300)
rc = norm(Ac * (splu(Ac; method = :multifrontal) \ bc) - bc) / norm(bc)
println("complex res=", round(rc, sigdigits = 2));
rc > 1.0e-8 && (ok = false);
Af = sparse(sprand(Float32, 400, 400, 10 / 400) + 5.0f0 * I);
bf = randn(Float32, 400);
rf = norm(Af * (splu(Af; method = :multifrontal) \ bf) - bf) / norm(bf)
println("float32 res=", round(rf, sigdigits = 2));
rf > 1.0f-3 && (ok = false);
println(ok ? "CORRECT_OK" : "CORRECT_FAIL")

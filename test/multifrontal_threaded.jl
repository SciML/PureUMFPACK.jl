# Threaded multifrontal driver (`threaded=true`): bit-identical to the serial path,
# workspace bounded by the thread count on wide elimination forests, and the same
# exception as the serial path on a singular pivot.
using PureUMFPACK
using PureUMFPACK: multifrontal_lu
using SparseArrays, LinearAlgebra, Random, Test

if Threads.nthreads() == 1
    # The threaded driver only engages with more than one thread.
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$(Base.active_project()) $(@__FILE__)`
    @test success(pipeline(cmd; stdout, stderr))
else
    include(joinpath(@__DIR__, "..", "benchmark", "matrices.jl"))
    Random.seed!(2024)
    threaded_lu(A; kw...) = multifrontal_lu(A; threaded = true, parallel_threshold = 0, kw...)

    @testset "threaded multifrontal ($(Threads.nthreads()) threads)" begin
        @testset "threaded == serial (identical factors & solution)" begin
            for A in (
                    poisson2d(24), poisson3d(8), poisson3d(12), randmat(800, 6),
                    arrowband(800, 4), blockdiag((poisson2d(6) for _ in 1:200)...),
                    sparse(ComplexF64.(poisson2d(20)) + im * I),
                )
                Fs = multifrontal_lu(A; check = false, threaded = false)
                Ft = threaded_lu(A; check = false)
                @test Fs.L.colptr == Ft.L.colptr && Fs.L.rowval == Ft.L.rowval
                @test Fs.U.colptr == Ft.U.colptr && Fs.U.rowval == Ft.U.rowval
                @test Fs.L.nzval == Ft.L.nzval && Fs.U.nzval == Ft.U.nzval
                @test Fs.p == Ft.p && Fs.q == Ft.q
                b = randn(eltype(A), size(A, 1))
                @test Fs \ b == Ft \ b
                @test norm(A * (Ft \ b) - b) / norm(b) <= 1.0e-8
            end
        end

        @testset "allocation stays proportional to serial on wide forests" begin
            # Each diagonal entry / block is its own elimination-tree root.
            for A in (
                    spdiagm(0 => collect(1.0:4000)),
                    blockdiag((poisson2d(6) for _ in 1:800)...),
                )
                multifrontal_lu(A; threaded = false)
                threaded_lu(A)
                serial = @allocated multifrontal_lu(A; threaded = false)
                threaded = @allocated threaded_lu(A)
                @test threaded <= 2 * serial
            end
        end

        @testset "singular pivot throws SingularException" begin
            D = spdiagm(0 => [i == 7 ? 0.0 : 1.0 for i in 1:64])
            @test_throws SingularException multifrontal_lu(D; threaded = false)
            @test_throws SingularException threaded_lu(D)
        end
    end
end

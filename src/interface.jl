# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2005-2023 Timothy A. Davis (UMFPACK, SuiteSparse) -- GPL-2.0-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# High-level factorization API mirroring SparseArrays' UMFPACK interface:
# the returned factorization satisfies  (Rs .* A)[p, q] == L * U.

"""
    PureLU

Pure-Julia sparse LU factorization. Satisfies `(F.Rs .* A)[F.p, F.q] == F.L * F.U`
with `L` unit lower triangular and `U` upper triangular — the same convention as
`SparseArrays.lu` (UMFPACK).
"""
struct PureLU{Tv, Ti <: Integer, Tr <: Real}
    L::SparseMatrixCSC{Tv, Ti}
    U::SparseMatrixCSC{Tv, Ti}
    p::Vector{Ti}
    q::Vector{Ti}
    Rs::Vector{Tr}
    A::SparseMatrixCSC{Tv, Ti}   # kept for residual / iterative refinement
end

Base.size(F::PureLU) = (size(F.L, 1), size(F.U, 2))
Base.size(F::PureLU, i::Integer) = size(F)[i]

"""
    splu(A; method=:gplu, ordering=:amd, tol=0.1, scale=SCALE_SUM, check=true) -> PureLU

Factorize sparse `A`.

- `method` ∈ (`:gplu`, `:multifrontal`). `:gplu` is the robust Gilbert–Peierls
  default (threshold partial pivoting, any unsymmetric matrix). `:multifrontal`
  is the supernodal BLAS-3 kernel — much faster on 3D / dense-front problems, with
  pivoting restricted to within each supernode (best for SPD / diagonally dominant
  / structurally symmetric systems); it ignores `ordering` (uses AMD + postorder).
- `ordering` ∈ (`:amd`, `:colamd`, `:natural`)  (`:gplu` only).
- `scale` ∈ (`SCALE_SUM`, `SCALE_MAX`, `SCALE_NONE`); `tol` is the pivot threshold.
"""
function splu(
        A::SparseMatrixCSC{Tv, Ti}; method::Symbol = :gplu, ordering::Symbol = :amd,
        tol::Real = 0.1, scale::ScaleKind = SCALE_SUM, check::Bool = true
    ) where {
        Tv, Ti <: Integer,
    }
    n = size(A, 2)
    size(A, 1) == n || throw(DimensionMismatch("splu requires a square matrix"))
    Rs = row_scaling(A, scale)
    As = scale == SCALE_NONE ? A : apply_row_scaling(A, Rs)
    if method === :multifrontal
        F = multifrontal_lu(As; check = check)
        return PureLU(F.L, F.U, F.p, F.q, Rs, A)
    elseif method !== :gplu
        throw(ArgumentError("unknown method $method"))
    end
    q = if ordering === :natural
        collect(Ti, 1:n)
    elseif ordering === :amd
        Ti.(amd_order_sym(A))
    elseif ordering === :colamd
        Ti.(colamd_order(A))
    else
        throw(ArgumentError("unknown ordering $ordering"))
    end
    F = gplu(As; q = q, tol = tol, check = check)
    return PureLU(F.L, F.U, F.p, F.q, Rs, A)
end

# One factorization solve: (Rs .* A)[p,q] = L U  =>  solve A x = b.
function _solve_factor(F::PureLU, b::AbstractVector)
    p = F.p
    q = F.q
    Rs = F.Rs
    c = similar(b, promote_type(eltype(F.U), eltype(b), eltype(Rs)))
    @inbounds for k in eachindex(p)
        c[k] = Rs[p[k]] * b[p[k]]
    end
    lsolve!(F.L, c)
    usolve!(F.U, c)
    x = similar(c)
    @inbounds for k in eachindex(q)
        x[q[k]] = c[k]
    end
    return x
end

"""
    solve(F::PureLU, b; refine=0) -> x

Solve `A x = b` using the factorization (handles row scaling Rs). `refine` extra
steps of iterative refinement (`x ← x + A⁻¹(b - A x)`) sharpen the result on
ill-conditioned systems — the same accuracy mechanism UMFPACK applies by default.
"""
function solve(F::PureLU, b::AbstractVector; refine::Integer = 0)
    x = _solve_factor(F, b)
    for _ in 1:refine
        r = b - F.A * x          # residual; sparse mat-vec from SparseArrays
        x = x + _solve_factor(F, r)
    end
    return x
end

Base.:\(F::PureLU, b::AbstractVector) = solve(F, b)

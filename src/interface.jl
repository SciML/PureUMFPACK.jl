# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2005-2023 Timothy A. Davis (UMFPACK, SuiteSparse) -- GPL-2.0-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# High-level factorization API mirroring SparseArrays' UMFPACK interface:
# the returned factorization satisfies  (Rs .* A)[p, q] == L * U.

"""
    PureLU

High-level pure-Julia sparse LU factorization returned by [`splu`](@ref). It
satisfies `(F.Rs .* A)[F.p, F.q] == F.L * F.U` with unit lower-triangular `L` and
upper-triangular `U`.

# Fields

- `L`: Unit lower-triangular sparse factor in pivot order.
- `U`: Upper-triangular sparse factor in pivot order.
- `p`: Row permutation used by the factorization.
- `q`: Column permutation used by the factorization.
- `Rs`: Scaling vector applied to rows before factorization.
- `A`: Original sparse matrix retained for iterative refinement.

# Interfaces

`PureLU` implements `Base.size` and `Base.\\`; `F \\ b` is equivalent to
`solve(F, b)` with `refine=0`.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> size(splu(sparse([2.0 1.0; 1.0 2.0])))
(2, 2)
```
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

Factorize a square sparse matrix with the high-level PureUMFPACK interface.

# Arguments

- `A`: Square sparse matrix to factorize.

# Keyword Arguments

- `method`: `:gplu` for robust Gilbert-Peierls LU on general unsymmetric matrices,
  or `:multifrontal` for the supernodal dense-front kernel. The latter uses AMD plus
  postordering and ignores `ordering`.
- `ordering`: `:amd`, `:colamd`, or `:natural` column order for `method=:gplu`.
- `tol`: Gilbert-Peierls pivot threshold in `[0, 1]`; see [`gplu`](@ref).
- `scale`: Row scaling policy: [`SCALE_SUM`](@ref), [`SCALE_MAX`](@ref), or
  [`SCALE_NONE`](@ref).
- `check`: Whether to throw `SingularException` for a singular pivot.

# Returns

A [`PureLU`](@ref) that supports `size`, [`solve`](@ref), and `\\`.

# Throws

- `DimensionMismatch`: `A` is not square.
- `ArgumentError`: `method` or `ordering` is not supported.
- `SingularException`: a pivot is singular and `check=true`.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> F = splu(sparse([2.0 1.0; 1.0 2.0]));

julia> F \\ [1.0, 0.0]
2-element Vector{Float64}:
  0.6666666666666666
 -0.3333333333333333
```
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

Solve `A * x = b` using a high-level [`PureLU`](@ref) factorization.

# Arguments

- `F`: Factorization returned by [`splu`](@ref).
- `b`: Right-hand-side vector with `length(b) == size(F, 1)`.

# Keyword Arguments

- `refine`: Number of iterative-refinement steps. Each step solves the residual
  equation with the existing factors.

# Returns

The solution vector in the original column order of `F.A`.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> solve(splu(sparse([2.0 1.0; 1.0 2.0])), [1.0, 0.0]; refine = 1)
2-element Vector{Float64}:
  0.6666666666666666
 -0.3333333333333333
```
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

# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Triangular solves on the CSC factors and the full sparse solve.
#
# After `_sortcols`, within each column row indices ascend, so the unit diagonal
# of L is the first entry of its column and the diagonal of U is the last.

# Solve L y = y in place (L unit lower triangular, CSC, diagonal = first entry).
function lsolve!(L::SparseMatrixCSC{Tv}, y::AbstractVector) where {Tv}
    Lp = _colptr(L)
    Li = rowvals(L)
    Lx = nonzeros(L)
    n = size(L, 2)
    @inbounds for j in 1:n
        yj = y[j]
        for p in (Lp[j] + 1):(Lp[j + 1] - 1)
            y[Li[p]] -= Lx[p] * yj
        end
    end
    return y
end

# Solve U y = y in place (U upper triangular, CSC, diagonal = last entry).
function usolve!(U::SparseMatrixCSC{Tv}, y::AbstractVector) where {Tv}
    Up = _colptr(U)
    Ui = rowvals(U)
    Ux = nonzeros(U)
    n = size(U, 2)
    @inbounds for j in n:-1:1
        pdiag = Up[j + 1] - 1
        yj = y[j] / Ux[pdiag]
        y[j] = yj
        for p in Up[j]:(pdiag - 1)
            y[Ui[p]] -= Ux[p] * yj
        end
    end
    return y
end

"""
    solve(F::GPLUFactorization, b) -> x

Solve the linear system `A * x = b` from a raw [`GPLUFactorization`](@ref).

# Arguments

- `F`: Factorization satisfying `A[F.p, F.q] == F.L * F.U`.
- `b`: Right-hand-side vector with `length(b) == size(F, 1)`.

# Returns

The solution vector `x` in the original column order of `A`.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> solve(gplu(sparse([2.0 1.0; 1.0 2.0])), [1.0, 0.0])
2-element Vector{Float64}:
  0.6666666666666666
 -0.3333333333333333
```
"""
function solve(F::GPLUFactorization, b::AbstractVector)
    p = F.p
    q = F.q
    c = b[p]
    lsolve!(F.L, c)
    usolve!(F.U, c)
    x = similar(b, promote_type(eltype(F.U), eltype(b)))
    @inbounds for j in eachindex(q)
        x[q[j]] = c[j]
    end
    return x
end

Base.:\(F::GPLUFactorization, b::AbstractVector) = solve(F, b)

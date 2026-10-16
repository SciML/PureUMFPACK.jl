# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2005-2023 Timothy A. Davis (UMFPACK, SuiteSparse) -- GPL-2.0-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Row scaling, matching UMFPACK's semantics: the returned factorization satisfies
#   (Rs .* A)[p, q] = L * U
# where `Rs .* A` scales row i of A by Rs[i].  UMFPACK's default is SCALE_SUM
# (Rs[i] = 1 / sum_j |A[i,j]|); SCALE_MAX uses the row's max abs value.  Scaling
# improves the effectiveness of threshold partial pivoting on badly scaled rows.

@enum ScaleKind SCALE_NONE SCALE_SUM SCALE_MAX

"""
    SCALE_NONE::PureUMFPACK.ScaleKind

Disable row scaling before factorization.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> A = sparse([1.0 0.0; 0.0 2.0]);

julia> row_scaling(A, SCALE_NONE);
```
"""
SCALE_NONE

"""
    SCALE_SUM::PureUMFPACK.ScaleKind

Scale each row by the inverse sum of absolute values in that row. This matches
UMFPACK's default row-scaling mode.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> A = sparse([1.0 2.0; 0.0 4.0]);

julia> row_scaling(A, SCALE_SUM);
```
"""
SCALE_SUM

"""
    SCALE_MAX::PureUMFPACK.ScaleKind

Scale each row by the inverse maximum absolute value in that row.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> A = sparse([1.0 2.0; 0.0 4.0]);

julia> row_scaling(A, SCALE_MAX);
```
"""
SCALE_MAX

"""
    row_scaling(A, kind) -> Rs::Vector

Compute the row scaling factors used by [`splu`](@ref).

# Arguments

- `A`: Sparse matrix whose rows are scaled.
- `kind`: Scaling policy: [`SCALE_NONE`](@ref), [`SCALE_SUM`](@ref), or
  [`SCALE_MAX`](@ref).

# Returns

A vector `Rs` such that row `i` of `A` is multiplied by `Rs[i]`. Empty or zero
rows receive a scale factor of one.

# Examples

```jldoctest
julia> using PureUMFPACK, SparseArrays

julia> row_scaling(sparse([1.0 2.0; 0.0 4.0]), SCALE_SUM)
2-element Vector{Float64}:
 0.3333333333333333
 0.25
```
"""
function row_scaling(A::SparseMatrixCSC{Tv, Ti}, kind::ScaleKind) where {Tv, Ti}
    n = size(A, 1)
    R = real(Tv)
    Rs = ones(R, n)
    kind == SCALE_NONE && return Rs
    acc = zeros(R, n)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    if kind == SCALE_SUM
        @inbounds for p in 1:nnz(A)
            acc[Ai[p]] += abs(Ax[p])
        end
    else # SCALE_MAX
        @inbounds for p in 1:nnz(A)
            i = Ai[p]
            v = abs(Ax[p])
            acc[i] = ifelse(v > acc[i], v, acc[i])
        end
    end
    @inbounds for i in 1:n
        Rs[i] = acc[i] > 0 ? inv(acc[i]) : one(R)
    end
    return Rs
end

# Apply row scaling in place to a copy: returns Diagonal(Rs) * A (same pattern).
function apply_row_scaling(A::SparseMatrixCSC{Tv, Ti}, Rs::AbstractVector) where {Tv, Ti}
    B = copy(A)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    @inbounds for p in 1:nnz(B)
        Bx[p] *= Rs[Bi[p]]
    end
    return B
end

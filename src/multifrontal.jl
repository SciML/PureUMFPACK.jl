# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2005-2023 Timothy A. Davis (UMFPACK, SuiteSparse) -- GPL-2.0-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Supernodal multifrontal LU — the UMFPACK-style dense-front algorithm.
#
# Factorizes V = A[qf,qf] (qf from `symbolic_mf`) over its structurally-symmetric
# pattern (A + Aᵀ).  Each supernode owns a contiguous block of pivot columns and
# a dense frontal matrix F = [ pivot | U-update ; L-update | Schur ].  Children's
# Schur (contribution) blocks assemble into the parent by extend-add, the pivot
# block is factored with LAPACK `getrf` (BLAS-3 `lu!`), and the off-diagonal
# blocks are updated with `trsm`/`gemm` (`ldiv!`/`rdiv!`/`mul!`).  This is where
# the dense-front flops run through optimized BLAS, closing the 3D gap GP-LU has.
#
# Pivoting is *restricted to within each supernode's pivot block* (rows c1..c2),
# so the nonzero structure is exactly the static symbolic prediction.  Both factors
# are written DIRECTLY into preallocated CSC arrays whose column pointers are
# counted up front from the symbolic structure: the dense-front numerics are
# scattered to their known CSC offsets via per-column cursors, no COO triplet
# streams, no global `sparse()` re-sort, and no row-permute of L.  L's row indices
# are stored in global-V coordinates during the loop and relabelled to factor order
# in one O(nnz) pass through `rowfac = invperm(prow)` at the end — the supernode's
# own pivots map to the contiguous block c1..c2 (sorted), update rows to their
# ancestor's factor rows.  U needs no row permutation (its rows are factor/elim
# indices already).  Processing supernodes in tree order (descendants first) makes
# every U column come out row-sorted (diagonal last) automatically, and L columns
# come out diagonal-first with the pivot block sorted — exactly the invariants the
# triangular solves rely on.
#
# This is stable for SPD / diagonally dominant / structurally symmetric systems
# (the 3D-Poisson-class targets); it does NOT do cross-front threshold (delayed)
# pivoting, so GP-LU remains the robust default for general unsymmetric stability.
#
# Allocation is O(1) large buffers, not O(nsuper) small ones: a single dense front
# workspace `Fbuf` (sized to the largest front) is reused for every supernode, and
# child contribution blocks live in one growable LIFO arena `cbval` (push on
# create, pop when the parent absorbs them).  This keeps garbage collection — which
# otherwise grew to 13–35% of runtime on large 3D problems — negligible.

# Extend-add a child's contribution block into the parent front `F`.  The CB is an
# m×m dense Schur block stored column-major in the shared arena `cbval` starting at
# 0-based offset `off`; its rows/cols are the global ids in `rows` (= the child's
# update set, aliased from `colstruct`).  Function barrier keeps this O(m²) hot loop
# fully typed for whatever concrete `F` (a Matrix view) is passed.
@inline function _extend_add!(
        F::AbstractMatrix{Tv}, loc::Vector{Int}, rows,
        cbval::Vector{Tv}, off::Int, m::Int
    ) where {Tv}
    @inbounds for s in 1:m
        cs = loc[rows[s]]
        base = off + (s - 1) * m
        for r in 1:m
            F[loc[rows[r]], cs] += cbval[base + r]
        end
    end
    return nothing
end

# Grow a 1-D arena to hold at least `need` elements (amortized doubling).
@inline function _ensure_cap!(v::Vector, need::Int)
    need > length(v) && resize!(v, max(need, 2 * length(v)))
    return nothing
end

# Count per-column nnz of L and U from the symbolic structure and build the CSC
# column pointers.  Returns (Lcolptr, Ucolptr) each length n+1, 1-based.
#   L column j (factor order): 1 diagonal + (c2-j) pivots below in supernode + nu
#                              shared update rows  (= 1 + (c2-j) + |colstruct[c2]|).
#   U column J: own-supernode upper part (rows cJ1..J) plus, for every descendant
#               supernode whose update set contains J, that supernode's whole pivot
#               column range.  Counted by scattering each supernode's pivot count
#               np onto its update columns, plus the triangular own part.
function _factor_colptrs(
        sstart::Vector{Int}, colstruct::Vector{Vector{Int}},
        n::Int, ::Type{Ti}
    ) where {Ti}
    nsuper = length(sstart) - 1
    Lcolptr = Vector{Ti}(undef, n + 1)
    Ucolptr = Vector{Ti}(undef, n + 1)
    ucount = zeros(Int, n)                     # per-column running nnz for U
    @inbounds for sk in 1:nsuper
        c1 = sstart[sk]
        c2 = sstart[sk + 1] - 1
        np = c2 - c1 + 1
        upd = colstruct[c2]
        nu = length(upd)
        for j in c1:c2
            Lcolptr[j + 1] = 1 + (c2 - j) + nu     # store as count, prefix-sum later
            ucount[j] += (j - c1 + 1)            # own upper-triangular pivot part
        end
        for t in 1:nu
            ucount[upd[t]] += np                 # this supernode fills np rows here
        end
    end
    Lcolptr[1] = 1
    @inbounds for j in 1:n
        Lcolptr[j + 1] += Lcolptr[j]
    end
    Ucolptr[1] = 1
    @inbounds for j in 1:n
        Ucolptr[j + 1] = Ucolptr[j] + ucount[j]
    end
    return Lcolptr, Ucolptr
end

"""
    multifrontal_lu(A::SparseMatrixCSC; q=nothing, tol=nothing, check=true) -> GPLUFactorization

Supernodal multifrontal LU. Returns the same `GPLUFactorization` (`A[p,q]==L*U`)
as [`gplu`](@ref), so it shares the triangular solves. `q` defaults to the AMD +
postorder ordering from [`symbolic_mf`](@ref).
"""
function multifrontal_lu(
        A::SparseMatrixCSC{Tv, Ti}; q = nothing, tol = nothing,
        check::Bool = true
    ) where {Tv, Ti <: Integer}
    n = size(A, 2)
    size(A, 1) == n || throw(DimensionMismatch("multifrontal_lu requires a square matrix"))
    S = q === nothing ? symbolic_mf(A) : symbolic_mf(A; q = q)
    qf = S.qf
    V = A[qf, qf]
    Vt = copy(transpose(V))
    Vp = getcolptr(V)
    Vi = rowvals(V)
    Vx = nonzeros(V)
    Vtp = getcolptr(Vt)
    Vti = rowvals(Vt)
    Vtx = nonzeros(Vt)

    sstart = S.sstart
    colstruct = S.colstruct
    childsupers = S.childsupers
    nsuper = length(sstart) - 1

    loc = zeros(Int, n)                      # global row/col -> front position
    prow = collect(1:n)                       # elim step -> global V-row (pivot)

    # One reusable dense front workspace (the top-left nf×nf block is used each
    # supernode) sized to the largest front — no per-supernode `zeros(nf,nf)`.
    maxnf = 0
    @inbounds for sk in 1:nsuper
        nfk = (sstart[sk + 1] - sstart[sk]) + length(colstruct[sstart[sk + 1] - 1])
        nfk > maxnf && (maxnf = nfk)
    end
    Fbuf = Matrix{Tv}(undef, maxnf, maxnf)

    # LIFO arena of child contribution blocks (each an nu×nu Schur block, stored
    # column-major).  The multifrontal assembly tree is processed in postorder, so
    # a supernode's children sit contiguously on top of the stack when it is reached
    # (classic multifrontal CB stack) — push on create, pop after absorbing.  Row
    # ids are not stored: a child's update set is `colstruct[its last column]`.
    cbval = Vector{Tv}(undef, max(64, maxnf * maxnf))
    cbtop = 0                                  # used length of the arena
    cboff = Vector{Int}(undef, nsuper)         # 0-based arena offset of each CB

    # Preallocated CSC factor storage; numerics scattered straight to these offsets.
    Lcolptr, Ucolptr = _factor_colptrs(sstart, colstruct, n, Ti)
    Lrowval = Vector{Ti}(undef, Lcolptr[n + 1] - 1)   # L rows in GLOBAL-V coords here
    Lnzval = Vector{Tv}(undef, Lcolptr[n + 1] - 1)
    Urowval = Vector{Ti}(undef, Ucolptr[n + 1] - 1)   # U rows in factor/elim coords
    Unzval = Vector{Tv}(undef, Ucolptr[n + 1] - 1)
    Lcur = Lcolptr[1:n]                              # write cursor per column
    Ucur = Ucolptr[1:n]

    @inbounds for sk in 1:nsuper
        c1 = sstart[sk]
        c2 = sstart[sk + 1] - 1
        np = c2 - c1 + 1
        upd = colstruct[c2]                   # rows > c2 (sorted), shared by block
        nu = length(upd)
        nf = np + nu

        for a in 1:np
            loc[c1 + a - 1] = a
        end
        for t in 1:nu
            loc[upd[t]] = np + t
        end

        F = view(Fbuf, 1:nf, 1:nf)            # reused workspace, zeroed for this front
        fill!(F, zero(Tv))

        # assemble original entries of V belonging to this front
        for a in 1:np
            j = c1 + a - 1
            for p in Vp[j]:(Vp[j + 1] - 1)        # column j: lower + diagonal
                i = Vi[p]
                if i >= j && loc[i] != 0
                    F[loc[i], a] += Vx[p]
                end
            end
            for p in Vtp[j]:(Vtp[j + 1] - 1)      # row j (=Vt[:,j]): strict upper
                k = Vti[p]
                if k > j && loc[k] != 0
                    F[a, loc[k]] += Vtx[p]
                end
            end
        end

        # extend-add children contribution blocks (LIFO: consume top-down so the
        # arena pops cleanly back to the level before this supernode's children).
        kids = childsupers[sk]
        for ci in length(kids):-1:1
            ck = kids[ci]
            crows = colstruct[sstart[ck + 1] - 1]
            m = length(crows)
            m == 0 && continue                # child produced no CB
            _extend_add!(F, loc, crows, cbval, cboff[ck], m)
            cbtop = cboff[ck]                 # pop child's CB off the arena
        end

        # ---- dense factorization of the pivot block (BLAS-3 getrf) ----
        # Partial pivoting restricted to the block's own rows (c1..c2).
        A11 = view(F, 1:np, 1:np)
        fac = lu!(A11, RowMaximum(); check = false)
        ip = fac.ipiv
        if check && fac.info != 0
            throw(SingularException(c1 + fac.info - 1))
        end

        if nu > 0
            R12 = view(F, 1:np, (np + 1):nf)
            for k in 1:np                     # apply block row swaps to U-update
                pk = ip[k]
                if pk != k
                    for col in 1:nu
                        R12[k, col], R12[pk, col] = R12[pk, col], R12[k, col]
                    end
                end
            end
            L21 = view(F, (np + 1):nf, 1:np)
            rdiv!(L21, UpperTriangular(A11))          # L21 = A21 * U11^{-1}
            ldiv!(UnitLowerTriangular(A11), R12)      # U12 = L11^{-1} * A12
            F22 = view(F, (np + 1):nf, (np + 1):nf)
            mul!(F22, L21, R12, -one(Tv), one(Tv))    # Schur := F22 - L21*U12
        end

        # record block row permutation (pivots live in {c1..c2})
        orig = _pivot_rows(c1, np, ip)
        for a in 1:np
            prow[c1 + a - 1] = orig[a]
        end

        # ---- scatter L (global-row coords) and U (factor coords) into CSC ----
        for a in 1:np
            gcol = c1 + a - 1
            # L column gcol: diagonal (orig[a], 1), pivots below, then update rows.
            lp = Lcur[gcol]
            Lrowval[lp] = orig[a]
            Lnzval[lp] = one(Tv)
            lp += 1
            for b in (a + 1):np
                Lrowval[lp] = orig[b]
                Lnzval[lp] = A11[b, a]
                lp += 1
            end
            for t in 1:nu
                Lrowval[lp] = upd[t]
                Lnzval[lp] = F[np + t, a]
                lp += 1
            end
            Lcur[gcol] = lp
            # U own upper-triangular part: column gcol receives rows c1..gcol.
            up = Ucur[gcol]
            for b in 1:a
                Urowval[up] = c1 + b - 1
                Unzval[up] = A11[b, a]
                up += 1
            end
            Ucur[gcol] = up
        end
        # U update part: each update column upd[t] receives rows c1..c2 (this whole
        # supernode's pivot range), values F[a, np+t].  Written in tree order so the
        # column stays row-sorted with its own diagonal last.
        for t in 1:nu
            J = upd[t]
            up = Ucur[J]
            for a in 1:np
                Urowval[up] = c1 + a - 1
                Unzval[up] = F[a, np + t]
                up += 1
            end
            Ucur[J] = up
        end

        # push this supernode's Schur complement (F22 = F[np+1:nf, np+1:nf]) onto
        # the arena for its parent to absorb.
        if nu > 0
            _ensure_cap!(cbval, cbtop + nu * nu)
            cboff[sk] = cbtop
            base = cbtop
            for s in 1:nu
                bs = base + (s - 1) * nu
                for r in 1:nu
                    cbval[bs + r] = F[np + r, np + s]
                end
            end
            cbtop += nu * nu
        end

        for a in 1:np
            loc[c1 + a - 1] = 0
        end
        for t in 1:nu
            loc[upd[t]] = 0
        end
    end

    # Relabel L's row indices from global-V coords to factor order in one O(nnz)
    # pass.  rowfac[g] = factor row of global V-row g (= invperm(prow)).  Each
    # supernode's own pivots map to the contiguous sorted block c1..c2 (so the L
    # pivot block stays sorted and the unit diagonal stays first); update rows map
    # to their owning ancestor's factor rows (> c2, off-diagonal, order-free).
    rowfac = invperm(prow)
    @inbounds for p in eachindex(Lrowval)
        Lrowval[p] = rowfac[Lrowval[p]]
    end

    L = SparseMatrixCSC(n, n, Lcolptr, Lrowval, Lnzval)
    U = SparseMatrixCSC(n, n, Ucolptr, Urowval, Unzval)

    p = qf[prow]                               # A[p,q] = L*U with q = qf
    pinv = invperm(p)
    return GPLUFactorization(L, U, p, collect(Ti, qf), pinv)
end

# Apply the in-block partial-pivot permutation `ip` (LAPACK getrf sequential row
# swaps) to the identity range c1:c2, giving the global V-row that ends up at each
# block position.  Returns a plain Vector for indexing.
@inline function _pivot_rows(c1::Int, np::Int, ip)
    orig = Vector{Int}(undef, np)
    @inbounds for a in 1:np
        orig[a] = c1 + a - 1
    end
    @inbounds for k in 1:np
        pk = ip[k]
        if pk != k
            orig[k], orig[pk] = orig[pk], orig[k]
        end
    end
    return orig
end

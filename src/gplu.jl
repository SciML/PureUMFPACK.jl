# Gilbert–Peierls left-looking sparse LU with partial (threshold) pivoting.
#
# This is the exact, unblocked numeric kernel: for each column k it solves the
# sparse lower-triangular system  L x = A(:, q[k])  using a depth-first
# reachability search to find the nonzero pattern of x (Gilbert & Peierls 1988),
# then chooses a pivot with threshold partial pivoting and appends one column to
# L and one to U.  The result satisfies  A[p, q] = L * U  with L unit lower
# triangular and U upper triangular — the same relation UMFPACK reports (with the
# row-scaling factor Rs handled separately by the caller).

"""
    GPLUFactorization

Raw output of [`gplu`](@ref): `A[p, q] == L * U`, `L` unit lower triangular,
`U` upper triangular.  `pinv` is the inverse row permutation (`pinv[p[k]] == k`).
"""
struct GPLUFactorization{Tv, Ti <: Integer}
    L::SparseMatrixCSC{Tv, Ti}
    U::SparseMatrixCSC{Tv, Ti}
    p::Vector{Ti}      # row permutation: row p[k] of A is pivot row k
    q::Vector{Ti}      # column permutation: column q[k] of A is factored at step k
    pinv::Vector{Ti}   # inverse of p
end

@inline function _grow2!(a::Vector, b::Vector)
    newcap = 2 * length(a)
    resize!(a, newcap)
    resize!(b, newcap)
    return nothing
end

# --- depth-first search over the partially built L, used by `reach!` ----------
#
# Stack-based DFS from start node `j0` in the directed graph of L (edge j -> i
# iff L[i,j] != 0).  A node `j` (an *original* row index) has outgoing edges only
# once it has become a pivot, i.e. when `pinv[j] >= 1`; then its neighbours are
# the row indices stored in column `pinv[j]` of the partial L.  Finished nodes
# are written to `xi[top..n]` in reverse-finish (topological) order.
@inline function _dfs!(j0::Ti, Lp, Li, pinv, top::Ti,
        xi::Vector{Ti}, stack::Vector{Ti}, pstack::Vector{Ti},
        marked::Vector{Bool}) where {Ti}
    head = 1
    stack[1] = j0
    @inbounds while head >= 1
        j = stack[head]
        J = pinv[j]
        if !marked[j]
            marked[j] = true
            pstack[head] = J >= 1 ? Lp[J] : zero(Ti)
        end
        done = true
        if J >= 1
            pend = Lp[J + 1] - 1
            p = pstack[head]
            while p <= pend
                i = Li[p]
                if !marked[i]
                    pstack[head] = p          # resume here after i's subtree
                    head += 1
                    stack[head] = i
                    done = false
                    break
                end
                p += 1
            end
        end
        if done
            head -= 1
            top -= one(Ti)
            xi[top] = j
        end
    end
    return top
end

# Reach(A(:,col)) in the graph of L, returned as the topologically ordered slice
# xi[top..n].  Leaves `marked` all-false on exit.
@inline function _reach!(Lp, Li, Ap, Ai, col::Ti, pinv,
        xi::Vector{Ti}, stack::Vector{Ti}, pstack::Vector{Ti},
        marked::Vector{Bool}, n::Ti) where {Ti}
    top = n + one(Ti)
    @inbounds for p in Ap[col]:(Ap[col + 1] - 1)
        b = Ai[p]
        if !marked[b]
            top = _dfs!(b, Lp, Li, pinv, top, xi, stack, pstack, marked)
        end
    end
    @inbounds for p in top:n
        marked[xi[p]] = false
    end
    return top
end

"""
    gplu(A::SparseMatrixCSC; q=1:size(A,2), tol=0.1, check=true, sort_factors=true)

Pure-Julia Gilbert–Peierls left-looking LU with threshold partial pivoting.
`q` is the (fill-reducing) column permutation; `tol ∈ [0,1]` is the pivot
threshold (1.0 = strict partial pivoting, 0.0 = pure diagonal-preferring).
`sort_factors=false` skips sorting row indices within columns — the factors still
solve correctly (the unit/diagonal entries stay positioned) but are not in the
canonical sorted CSC form.
"""
function gplu(
        A::SparseMatrixCSC{Tv, Ti}; q::AbstractVector{<:Integer} = Base.OneTo(size(A, 2)),
        tol::Real = 0.1, check::Bool = true, sort_factors::Bool = true) where {
        Tv, Ti <: Integer}
    n = size(A, 2)
    size(A, 1) == n || throw(DimensionMismatch("gplu requires a square matrix"))
    qv = collect(Ti, q)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    Ax = nonzeros(A)

    # workspaces
    x = zeros(Tv, n)
    xi = Vector{Ti}(undef, n)
    stack = Vector{Ti}(undef, n)
    pstack = Vector{Ti}(undef, n)
    marked = fill(false, n)
    pinv = zeros(Ti, n)               # 0 == not yet a pivot

    # L/U are built into preallocated arrays that grow geometrically; the DFS
    # reads L (Lp, Li) in place as it is filled.
    Lp = Vector{Ti}(undef, n + 1)
    Lp[1] = 1
    Up = Vector{Ti}(undef, n + 1)
    Up[1] = 1
    cap = max(8 * nnz(A), 16)
    Li = Vector{Ti}(undef, cap)
    Lx = Vector{Tv}(undef, cap)
    lnz = 0
    Ui = Vector{Ti}(undef, cap)
    Ux = Vector{Tv}(undef, cap)
    unz = 0

    tolv = real(Tv)(tol)               # pivot threshold is a real magnitude
    @inbounds for k in 1:n
        col = qv[k]
        top = _reach!(Lp, Li, Ap, Ai, col, pinv, xi, stack, pstack, marked, Ti(n))

        # scatter A(:,col) into x over the reach (reach is zero coming in)
        for p in top:n
            x[xi[p]] = zero(Tv)
        end
        for p in Ap[col]:(Ap[col + 1] - 1)
            x[Ai[p]] = Ax[p]
        end
        # forward solve L x = A(:,col), L has unit diagonal (stored first)
        for px in top:n
            j = xi[px]
            J = pinv[j]
            J < 1 && continue
            xj = x[j]
            for p in (Lp[J] + 1):(Lp[J + 1] - 1)
                x[Li[p]] -= Lx[p] * xj
            end
        end

        # pivot selection: U entries (already-pivot rows) + largest unpivoted
        ipiv = zero(Ti)
        a = -one(real(Tv))
        for px in top:n
            i = xi[px]
            pii = pinv[i]
            if pii == 0
                t = abs(x[i])
                if t > a
                    a = t
                    ipiv = i
                end
            else
                unz += 1
                unz > length(Ui) && _grow2!(Ui, Ux)
                Ui[unz] = pii
                Ux[unz] = x[i]
            end
        end
        if ipiv == 0 || a <= 0
            check && throw(SingularException(k))
            ipiv = ipiv == 0 ? Ti(col) : ipiv   # best-effort continue
        end
        # threshold partial pivoting: prefer the diagonal column if big enough
        if pinv[col] == 0 && abs(x[col]) >= tolv * a
            ipiv = Ti(col)
        end
        pivot = x[ipiv]

        unz += 1
        unz > length(Ui) && _grow2!(Ui, Ux)
        Ui[unz] = k
        Ux[unz] = pivot            # U diagonal, stored last
        Up[k + 1] = unz + 1
        pinv[ipiv] = k

        lnz += 1
        lnz > length(Li) && _grow2!(Li, Lx)
        Li[lnz] = ipiv
        Lx[lnz] = one(Tv)       # L unit diagonal, stored first
        for px in top:n
            i = xi[px]
            if pinv[i] == 0
                lnz += 1
                lnz > length(Li) && _grow2!(Li, Lx)
                Li[lnz] = i
                Lx[lnz] = x[i] / pivot
            end
            x[i] = zero(Tv)                      # clear workspace for next column
        end
        Lp[k + 1] = lnz + 1
    end

    resize!(Li, lnz)
    resize!(Lx, lnz)
    resize!(Ui, unz)
    resize!(Ux, unz)

    # map L's stored (original) row indices into pivot order
    @inbounds for t in eachindex(Li)
        Li[t] = pinv[Li[t]]
    end

    L = SparseMatrixCSC(n, n, Lp, Li, Lx)
    U = SparseMatrixCSC(n, n, Up, Ui, Ux)
    if sort_factors
        L = _sortcols(L)
        U = _sortcols(U)
    end

    p = invperm(pinv)
    return GPLUFactorization(L, U, p, qv, copy(pinv))
end

# Insertion sort of the parallel (key, value) arrays a[lo:hi]/b[lo:hi] by key.
# Allocation-free; ideal for the short columns typical of sparse factors.
@inline function _isort!(a, b, lo::Integer, hi::Integer)
    @inbounds for i in (lo + 1):hi
        ka = a[i]
        kb = b[i]
        j = i - 1
        while j >= lo && a[j] > ka
            a[j + 1] = a[j]
            b[j + 1] = b[j]
            j -= 1
        end
        a[j + 1] = ka
        b[j + 1] = kb
    end
    return nothing
end

# Bottom-up merge sort of a[lo:hi]/b[lo:hi] by key.  Uses two 1-based scratch
# pairs (s1a/s1b and s2a/s2b, each length >= len) so the merge is offset-free and
# obviously correct.  Allocation-free given the scratch; stable.
function _msort!(a, b, lo::Integer, hi::Integer, s1a, s1b, s2a, s2b)
    len = hi - lo + 1
    @inbounds for t in 1:len            # gather slice into scratch pair 1 (1-based)
        s1a[t] = a[lo + t - 1]
        s1b[t] = b[lo + t - 1]
    end
    cura, curb, otha, othb = s1a, s1b, s2a, s2b
    width = 1
    @inbounds while width < len
        i = 1
        while i <= len
            l = i
            m = min(i + width, len + 1)      # start of right run (1-based, exclusive-left)
            r = min(i + 2width, len + 1)     # end (exclusive)
            p = l
            q = m
            k = l
            while p < m && q < r
                if cura[p] <= cura[q]
                    otha[k] = cura[p]
                    othb[k] = curb[p]
                    p += 1
                else
                    otha[k] = cura[q]
                    othb[k] = curb[q]
                    q += 1
                end
                k += 1
            end
            while p < m
                otha[k] = cura[p]
                othb[k] = curb[p]
                p += 1
                k += 1
            end
            while q < r
                otha[k] = cura[q]
                othb[k] = curb[q]
                q += 1
                k += 1
            end
            i += 2width
        end
        cura, curb, otha, othb = otha, othb, cura, curb   # swap
        width *= 2
    end
    @inbounds for t in 1:len            # scatter sorted result back
        a[lo + t - 1] = cura[t]
        b[lo + t - 1] = curb[t]
    end
    return nothing
end

# Sort row indices within each column of a CSC built out of order (no duplicates).
# Genuinely allocation-free: scratch sized to the longest column allocated once,
# hybrid insertion (short columns) / merge sort (long columns), no per-call alloc.
function _sortcols(S::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    cp = getcolptr(S)
    ri = rowvals(S)
    nz = nonzeros(S)
    n = size(S, 2)
    maxlen = 0
    @inbounds for j in 1:n
        maxlen = max(maxlen, cp[j + 1] - cp[j])
    end
    s1a = Vector{Ti}(undef, maxlen)
    s1b = Vector{Tv}(undef, maxlen)
    s2a = Vector{Ti}(undef, maxlen)
    s2b = Vector{Tv}(undef, maxlen)
    @inbounds for j in 1:n
        lo = cp[j]
        hi = cp[j + 1] - 1
        len = hi - lo + 1
        len <= 1 && continue
        sorted = true
        for p in lo:(hi - 1)
            if ri[p] > ri[p + 1]
                sorted = false
                break
            end
        end
        sorted && continue
        if len <= 32
            _isort!(ri, nz, lo, hi)
        else
            _msort!(ri, nz, lo, hi, s1a, s1b, s2a, s2b)
        end
    end
    return S
end

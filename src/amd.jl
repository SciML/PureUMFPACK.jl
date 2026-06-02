# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2006 Timothy A. Davis (CSparse, SuiteSparse) -- LGPL-2.1-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Pure-Julia Approximate Minimum Degree (symmetric, on the pattern of A + Aᵀ).
#
# Faithful port of Tim Davis's CSparse `cs_amd` (order = 1).  Translation strategy:
# index *values* stay 0-based exactly as in the C (so the -1 nil sentinel and the
# CS_FLIP(i) = -i-2 arithmetic are unchanged); whenever a non-negative value is
# used to index a Julia array we add 1.  Validated against the AMD package: it
# reproduces AMD-quality fill (see test/amd_test.jl).

@inline _flip(x::Int) = -x - 2
@inline _unflip(x::Int) = x < 0 ? _flip(x) : x

# w-mark management: keep marks strictly above `mark`; reset on overflow.
@inline function _wclear(mark::Int, lemax::Int, w::Vector{Int}, n::Int)
    if mark < 2 || (mark + lemax < 0)
        @inbounds for k in 0:(n - 1)
            w[k + 1] != 0 && (w[k + 1] = 1)
        end
        mark = 2
    end
    return mark
end

# Postorder a tree (assembly tree) rooted at node j, numbering from k upward.
function _tdfs(
        j::Int, k::Int, head::Vector{Int}, next::Vector{Int},
        post::Vector{Int}, stack::Vector{Int}
    )
    top = 0
    @inbounds stack[1] = j
    @inbounds while top >= 0
        p = stack[top + 1]
        i = head[p + 1]
        if i == -1
            top -= 1
            post[k + 1] = p
            k += 1
        else
            head[p + 1] = next[i + 1]
            top += 1
            stack[top + 1] = i
        end
    end
    return k
end

# Build the 0-based CSC pattern of `S` with the diagonal removed.
# Returns (cp, ci, cnz) where cp has length n+1 (cp[j+1] = 0-based start of col j).
function _drop_diag_0based(S::SparseMatrixCSC)
    n = size(S, 2)
    Sp = getcolptr(S)
    Si = rowvals(S)
    cnz = 0
    @inbounds for j in 1:n, p in Sp[j]:(Sp[j + 1] - 1)
        Si[p] != j && (cnz += 1)
    end
    ci = Vector{Int}(undef, cnz)
    cp = Vector{Int}(undef, n + 1)
    cp[1] = 0
    q = 0
    @inbounds for j in 1:n
        for p in Sp[j]:(Sp[j + 1] - 1)
            i = Si[p]
            if i != j
                ci[q + 1] = i - 1               # 0-based row index
                q += 1
            end
        end
        cp[j + 1] = q
    end
    return cp, ci, cnz
end

function _patmat(A::SparseMatrixCSC)
    return SparseMatrixCSC(
        size(A, 1), size(A, 2), copy(getcolptr(A)), copy(rowvals(A)), ones(nnz(A))
    )
end

# pattern of (A + Aᵀ), diagonal dropped, 0-based
function _sym_pattern(A::SparseMatrixCSC)
    Pat = _patmat(A)
    return _drop_diag_0based(Pat + copy(transpose(Pat)))
end

# pattern of (AᵀA) — the column-intersection graph — with dense rows of A dropped,
# diagonal dropped, 0-based.  This is the CSparse cs_amd(order=2) construction.
function _ata_pattern(A::SparseMatrixCSC; dense_param::Real = 10.0)
    n = size(A, 2)
    Pat = _patmat(A)
    AT = copy(transpose(Pat))                 # AT[:,j] = row j of A
    ATp = getcolptr(AT)
    densethr = floor(Int, min(n - 2, max(16.0, dense_param * sqrt(n))))
    keep = trues(size(AT, 2))
    @inbounds for j in 1:size(AT, 2)
        (ATp[j + 1] - ATp[j]) > densethr && (keep[j] = false)
    end
    AT2 = AT[:, keep]                          # drop dense rows of A
    C = AT2 * copy(transpose(AT2))            # AᵀA pattern (values = intersection counts)
    return _drop_diag_0based(C)
end

"""
    amd_order_sym(A::SparseMatrixCSC) -> Vector{Int}

Approximate minimum degree permutation `p` (1-based) for the symmetric pattern of
`A + Aᵀ`.  `(A+Aᵀ)[p,p]` has a sparse Cholesky/LU factor.
"""
function amd_order_sym(A::SparseMatrixCSC; dense::Real = 10.0)
    n = size(A, 2)
    n == size(A, 1) || throw(DimensionMismatch("amd requires a square matrix"))
    n == 0 && return Int[]
    n == 1 && return [1]
    cp, ci0, cnz = _sym_pattern(A)
    return _amd_core(n, cp, ci0, cnz; dense = dense)
end

"""
    colamd_order(A::SparseMatrixCSC) -> Vector{Int}

Column fill-reducing ordering for unsymmetric LU: AMD on the column-intersection
graph (pattern of AᵀA with dense rows dropped), following CSparse `cs_amd(order=2)`.
Returns a permutation of the columns of `A` suitable as the `q` argument to `gplu`.
"""
function colamd_order(A::SparseMatrixCSC; dense::Real = 10.0)
    n = size(A, 2)
    n <= 1 && return collect(1:n)
    cp, ci0, cnz = _ata_pattern(A; dense_param = dense)
    return _amd_core(n, cp, ci0, cnz; dense = dense)
end

# Core quotient-graph AMD over a symmetric 0-based pattern (cp, ci0, cnz).
function _amd_core(n::Int, cp::Vector{Int}, ci0::Vector{Int}, cnz::Int; dense::Real = 10.0)
    # over-allocate the working adjacency array (CSparse: t = cnz + cnz/5 + 2n)
    t = cnz + cnz ÷ 5 + 2 * n
    Ci = Vector{Int}(undef, max(t, 1))
    @inbounds copyto!(Ci, 1, ci0, 1, cnz)

    dthresh = max(16.0, dense * sqrt(n))
    densethr = Int(min(n - 2, floor(Int, dthresh)))

    # workspace; C indices 0..n -> Julia 1..n+1
    len = Vector{Int}(undef, n + 1)
    nv = Vector{Int}(undef, n + 1)
    nxt = Vector{Int}(undef, n + 1)   # 'next'
    head = Vector{Int}(undef, n + 1)
    elen = Vector{Int}(undef, n + 1)
    degree = Vector{Int}(undef, n + 1)
    w = Vector{Int}(undef, n + 1)
    hhead = Vector{Int}(undef, n + 1)
    last = Vector{Int}(undef, n + 1)
    P = Vector{Int}(undef, n + 1)
    Cp = Vector{Int}(undef, n + 1)

    @inbounds for k in 0:(n - 1)
        len[k + 1] = cp[k + 2] - cp[k + 1]
        Cp[k + 1] = cp[k + 1]
    end
    @inbounds Cp[n + 1] = cp[n + 1]
    @inbounds len[n + 1] = 0

    nzmax = length(Ci)

    @inbounds for i in 0:n
        head[i + 1] = -1
        last[i + 1] = -1
        nxt[i + 1] = -1
        hhead[i + 1] = -1
        nv[i + 1] = 1
        w[i + 1] = 1
        elen[i + 1] = 0
        degree[i + 1] = len[i + 1]
    end
    mark = _wclear(0, 0, w, n)
    @inbounds begin
        elen[n + 1] = -2
        Cp[n + 1] = -1
        w[n + 1] = 0
    end

    nel = 0
    @inbounds for i in 0:(n - 1)
        d = degree[i + 1]
        if d == 0
            elen[i + 1] = -2
            nel += 1
            Cp[i + 1] = -1
            w[i + 1] = 0
        elseif d > densethr
            nv[i + 1] = 0
            elen[i + 1] = -1
            nel += 1
            Cp[i + 1] = _flip(n)
            nv[n + 1] += 1
        else
            if head[d + 1] != -1
                last[head[d + 1] + 1] = i
            end
            nxt[i + 1] = head[d + 1]
            head[d + 1] = i
        end
    end

    mindeg = 0
    lemax = 0
    cnz_cur = cnz

    @inbounds while nel < n
        # select node of minimum approximate degree
        k = -1
        while mindeg < n
            k = head[mindeg + 1]
            k != -1 && break
            mindeg += 1
        end
        if nxt[k + 1] != -1
            last[nxt[k + 1] + 1] = -1
        end
        head[mindeg + 1] = nxt[k + 1]
        elenk = elen[k + 1]
        nvk = nv[k + 1]
        nel += nvk

        # garbage collection
        if elenk > 0 && cnz_cur + mindeg >= nzmax
            for j in 0:(n - 1)
                p = Cp[j + 1]
                if p >= 0
                    Cp[j + 1] = Ci[p + 1]
                    Ci[p + 1] = _flip(j)
                end
            end
            q = 0
            p = 0
            while p < cnz_cur
                j = _flip(Ci[p + 1])
                p += 1
                if j >= 0
                    Ci[q + 1] = Cp[j + 1]
                    Cp[j + 1] = q
                    q += 1
                    for _k3 in 0:(len[j + 1] - 2)
                        Ci[q + 1] = Ci[p + 1]
                        q += 1
                        p += 1
                    end
                end
            end
            cnz_cur = q
        end

        # construct new element from k
        dk = 0
        nv[k + 1] = -nvk
        p = Cp[k + 1]
        pk1 = elenk == 0 ? p : cnz_cur
        pk2 = pk1
        for k1 in 1:(elenk + 1)
            if k1 > elenk
                e = k
                pj = p
                ln = len[k + 1] - elenk
            else
                e = Ci[p + 1]
                p += 1
                pj = Cp[e + 1]
                ln = len[e + 1]
            end
            for _k2 in 1:ln
                i = Ci[pj + 1]
                pj += 1
                nvi = nv[i + 1]
                nvi <= 0 && continue
                dk += nvi
                nv[i + 1] = -nvi
                Ci[pk2 + 1] = i
                pk2 += 1
                if nxt[i + 1] != -1
                    last[nxt[i + 1] + 1] = last[i + 1]
                end
                if last[i + 1] != -1
                    nxt[last[i + 1] + 1] = nxt[i + 1]
                else
                    head[degree[i + 1] + 1] = nxt[i + 1]
                end
            end
            if e != k
                Cp[e + 1] = _flip(k)
                w[e + 1] = 0
            end
        end
        if elenk != 0
            cnz_cur = pk2
        end
        degree[k + 1] = dk
        Cp[k + 1] = pk1
        len[k + 1] = pk2 - pk1
        elen[k + 1] = -2

        # find set differences (scan 1)
        mark = _wclear(mark, lemax, w, n)
        for pk in pk1:(pk2 - 1)
            i = Ci[pk + 1]
            eln = elen[i + 1]
            eln <= 0 && continue
            nvi = -nv[i + 1]
            wnvi = mark - nvi
            for p2 in Cp[i + 1]:(Cp[i + 1] + eln - 1)
                e = Ci[p2 + 1]
                if w[e + 1] >= mark
                    w[e + 1] -= nvi
                elseif w[e + 1] != 0
                    w[e + 1] = degree[e + 1] + wnvi
                end
            end
        end

        # degree update (scan 2)
        for pk in pk1:(pk2 - 1)
            i = Ci[pk + 1]
            p1 = Cp[i + 1]
            p2 = p1 + elen[i + 1] - 1
            pn = p1
            h = 0
            d = 0
            p3loop = p1
            while p3loop <= p2
                e = Ci[p3loop + 1]
                if w[e + 1] != 0
                    dext = w[e + 1] - mark
                    if dext > 0
                        d += dext
                        Ci[pn + 1] = e
                        pn += 1
                        h += e
                    else
                        Cp[e + 1] = _flip(k)
                        w[e + 1] = 0   # aggressive absorption
                    end
                end
                p3loop += 1
            end
            elen[i + 1] = pn - p1 + 1
            p3 = pn
            p4 = p1 + len[i + 1]
            for p5 in (p2 + 1):(p4 - 1)
                j = Ci[p5 + 1]
                nvj = nv[j + 1]
                nvj <= 0 && continue
                d += nvj
                Ci[pn + 1] = j
                pn += 1
                h += j
            end
            if d == 0
                Cp[i + 1] = _flip(k)
                nvi = -nv[i + 1]
                dk -= nvi
                nvk += nvi
                nel += nvi
                nv[i + 1] = 0
                elen[i + 1] = -1
            else
                degree[i + 1] = min(degree[i + 1], d)
                Ci[pn + 1] = Ci[p3 + 1]
                Ci[p3 + 1] = Ci[p1 + 1]
                Ci[p1 + 1] = k
                len[i + 1] = pn - p1 + 1
                h = mod(h, n)
                nxt[i + 1] = hhead[h + 1]
                hhead[h + 1] = i
                last[i + 1] = h
            end
        end
        degree[k + 1] = dk
        lemax = max(lemax, dk)
        mark = _wclear(mark + lemax, lemax, w, n)

        # supervariable detection
        for pk in pk1:(pk2 - 1)
            i = Ci[pk + 1]
            nv[i + 1] >= 0 && continue
            h = last[i + 1]
            i = hhead[h + 1]
            hhead[h + 1] = -1
            while i != -1 && nxt[i + 1] != -1
                ln = len[i + 1]
                eln = elen[i + 1]
                for p in (Cp[i + 1] + 1):(Cp[i + 1] + ln - 1)
                    w[Ci[p + 1] + 1] = mark
                end
                jlast = i
                j = nxt[i + 1]
                while j != -1
                    ok = (len[j + 1] == ln) && (elen[j + 1] == eln)
                    p = Cp[j + 1] + 1
                    while ok && p <= Cp[j + 1] + ln - 1
                        w[Ci[p + 1] + 1] != mark && (ok = false)
                        p += 1
                    end
                    if ok
                        Cp[j + 1] = _flip(i)
                        nv[i + 1] += nv[j + 1]
                        nv[j + 1] = 0
                        elen[j + 1] = -1
                        j = nxt[j + 1]
                        nxt[jlast + 1] = j
                    else
                        jlast = j
                        j = nxt[j + 1]
                    end
                end
                i = nxt[i + 1]
                mark += 1
            end
        end

        # finalize new element k, restore degree lists
        p = pk1
        for pk in pk1:(pk2 - 1)
            i = Ci[pk + 1]
            nvi = -nv[i + 1]
            nvi <= 0 && continue
            nv[i + 1] = nvi
            d = degree[i + 1] + dk - nvi
            d = min(d, n - nel - nvi)
            if head[d + 1] != -1
                last[head[d + 1] + 1] = i
            end
            nxt[i + 1] = head[d + 1]
            last[i + 1] = -1
            head[d + 1] = i
            mindeg = min(mindeg, d)
            degree[i + 1] = d
            Ci[p + 1] = i
            p += 1
        end
        nv[k + 1] = nvk
        len[k + 1] = p - pk1
        if len[k + 1] == 0
            Cp[k + 1] = -1
            w[k + 1] = 0
        end
        if elenk != 0
            cnz_cur = p
        end
    end

    # postorder the assembly tree -> permutation
    @inbounds for i in 0:(n - 1)
        Cp[i + 1] = _flip(Cp[i + 1])
    end
    @inbounds for j in 0:n
        head[j + 1] = -1
    end
    @inbounds for j in n:-1:0
        nv[j + 1] > 0 && continue
        nxt[j + 1] = head[Cp[j + 1] + 1]
        head[Cp[j + 1] + 1] = j
    end
    @inbounds for e in n:-1:0
        nv[e + 1] <= 0 && continue
        if Cp[e + 1] != -1
            nxt[e + 1] = head[Cp[e + 1] + 1]
            head[Cp[e + 1] + 1] = e
        end
    end
    k = 0
    @inbounds for i in 0:n
        if Cp[i + 1] == -1
            k = _tdfs(i, k, head, nxt, P, w)
        end
    end

    # P[1..n] holds 0-based permutation; convert to 1-based
    perm = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        perm[i] = P[i] + 1
    end
    return perm
end

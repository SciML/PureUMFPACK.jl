# Symbolic analysis for the supernodal multifrontal factorization.
#
# Pipeline (all on the symmetric pattern of A + Aᵀ):
#   1. AMD fill-reducing order  q                       (src/amd.jl)
#   2. elimination tree of A[q,q]  +  postorder         -> final order qf = q[post]
#   3. column structure of L (each column's row indices below the diagonal)
#   4. fundamental-supernode amalgamation
# The result feeds the numeric multifrontal kernel in src/multifrontal.jl.

# Symmetric pattern of A + Aᵀ as a 1-based CSC (sorted rows, diagonal dropped).
function _sym_csc1(A::SparseMatrixCSC)
    Pat = _patmat(A)
    S = Pat + copy(transpose(Pat))            # sorted CSC
    n = size(S, 2)
    Sp = getcolptr(S)
    Si = rowvals(S)
    cp = Vector{Int}(undef, n + 1)
    cp[1] = 1
    ri = Int[]
    sizehint!(ri, nnz(S))
    @inbounds for j in 1:n
        for p in Sp[j]:(Sp[j + 1] - 1)
            i = Si[p]
            i != j && push!(ri, i)
        end
        cp[j + 1] = length(ri) + 1
    end
    return cp, ri
end

# Symmetric permutation of an already-built symmetric pattern (cp, ri) by `perm`
# (1-based; new column j corresponds to old column perm[j]).  Returns the pattern
# of B[perm,perm] as a 1-based CSC with sorted rows and diagonal dropped — i.e.
# exactly what `_sym_csc1` would return for the permuted matrix, but derived from
# the existing pattern instead of rebuilding/symmetrizing a SparseMatrixCSC slice.
function _permute_sym_csc1(cp::Vector{Int}, ri::Vector{Int}, perm::Vector{Int}, n::Int)
    pinv = invperm(perm)                       # old row -> new row
    cpN = Vector{Int}(undef, n + 1)
    nz = cp[n + 1] - 1
    riN = Vector{Int}(undef, nz)
    # column sizes are preserved under symmetric permutation: |col perm[j]| -> col j
    cpN[1] = 1
    @inbounds for j in 1:n
        oj = perm[j]
        cpN[j + 1] = cpN[j] + (cp[oj + 1] - cp[oj])
    end
    @inbounds for j in 1:n
        oj = perm[j]
        dst = cpN[j] - 1
        for p in cp[oj]:(cp[oj + 1] - 1)
            dst += 1
            riN[dst] = pinv[ri[p]]
        end
        lo = cpN[j]
        hi = cpN[j + 1] - 1
        hi > lo && sort!(view(riN, lo:hi))     # restore ascending rows within column
    end
    return cpN, riN
end

# Elimination tree of a symmetric pattern (Liu / CSparse cs_etree).  parent[k]=0
# marks a root.  nil is encoded as 0 throughout (1-based indices).
function _etree(cp::Vector{Int}, ri::Vector{Int}, n::Int)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    @inbounds for k in 1:n
        for p in cp[k]:(cp[k + 1] - 1)
            i = ri[p]
            while i != 0 && i < k
                inext = ancestor[i]
                ancestor[i] = k
                inext == 0 && (parent[i] = k)
                i = inext
            end
        end
    end
    return parent
end

# Depth-first postorder numbering of the forest given by `parent` (CSparse cs_post
# + cs_tdfs), using child lists built so siblings come out in ascending order.
function _postorder(parent::Vector{Int})
    n = length(parent)
    head = zeros(Int, n)
    nextc = zeros(Int, n)
    @inbounds for j in n:-1:1
        p = parent[j]
        p != 0 && (nextc[j] = head[p]; head[p] = j)
    end
    post = zeros(Int, n)
    stack = zeros(Int, n)
    k = 1
    @inbounds for j in 1:n
        parent[j] != 0 && continue        # start only at roots
        top = 1
        stack[1] = j
        while top >= 1
            p = stack[top]
            i = head[p]
            if i == 0
                top -= 1
                post[k] = p
                k += 1
            else
                head[p] = nextc[i]
                top += 1
                stack[top] = i
            end
        end
    end
    return post
end

# Column structure of L for a *postordered* symmetric pattern (parent[j] > j,
# every node's descendants precede it).  colstruct[j] = sorted row indices i > j
# in column j of L.  Uses the classic union rule: col j = A(:,j)_{i>j} ∪ over
# children c of j of (colstruct[c] \ {j}).
function _col_structure(cp::Vector{Int}, ri::Vector{Int}, parent::Vector{Int}, n::Int)
    colstruct = Vector{Vector{Int}}(undef, n)
    marker = zeros(Int, n)
    head = zeros(Int, n)
    nextc = zeros(Int, n)
    @inbounds for j in n:-1:1
        p = parent[j]
        p != 0 && (nextc[j] = head[p]; head[p] = j)
    end
    scratch = Vector{Int}(undef, n)        # reused across columns; no per-col growth
    @inbounds for j in 1:n
        cnt = 0
        marker[j] = j                      # block the diagonal
        for p in cp[j]:(cp[j + 1] - 1)
            i = ri[p]
            if i > j && marker[i] != j
                marker[i] = j
                cnt += 1
                scratch[cnt] = i
            end
        end
        c = head[j]
        while c != 0
            for i in colstruct[c]
                if i > j && marker[i] != j
                    marker[i] = j
                    cnt += 1
                    scratch[cnt] = i
                end
            end
            c = nextc[c]
        end
        col = Vector{Int}(undef, cnt)      # exact size; single allocation per column
        copyto!(col, 1, scratch, 1, cnt)
        sort!(col)
        colstruct[j] = col
    end
    return colstruct
end

# Supernode partition of a postordered tree.  Column j joins the previous
# supernode iff j-1 is j's only child and the structures nest closely enough.
# For a *fundamental* supernode the nesting is exact (colcount[j-1] == colcount[j]
# + 1, no extra entries); `relax` allows merging when the merge would introduce at
# most `relax` extra explicit (structural-zero) entries in column j-1, namely
# e = (1 + colcount[j]) - colcount[j-1] (always >= 0 here since column j-1's
# structure is a subset of {j} ∪ colstruct[j] when j-1 is j's only child).  With
# `relax == 0` this reduces exactly to the fundamental partition.
#
# When a relaxed (non-fundamental) merge is accepted, the per-column L structure
# of the interior columns is expanded in place so every column c of a supernode
# [c1..c2] has colstruct[c] == {c+1, …, c2} ∪ colstruct[c2] — the dense pivot
# block plus the shared update set the numeric front actually fills.  This keeps
# `predicted_fill`, `_factor_colptrs`, and the kernel's scatter in exact agreement:
# all the extra explicit entries are accounted for in the static structure.
#
# Returns (sstart, super_of, childsupers):
#   sstart[k]:sstart[k+1]-1  = columns of supernode k
#   childsupers[k]           = child supernodes of k (their CBs assemble into k)
function _supernodes(parent::Vector{Int}, colstruct::Vector{Vector{Int}}, n::Int;
        relax::Int = 0)
    colcount = [length(colstruct[j]) for j in 1:n]
    nchild = zeros(Int, n)
    @inbounds for j in 1:n
        p = parent[j]
        p != 0 && (nchild[p] += 1)
    end
    super_of = zeros(Int, n)
    sstart = Int[]
    nsuper = 0
    relaxed = false                       # any non-fundamental merge accepted?
    @inbounds for j in 1:n
        chain = j > 1 && parent[j - 1] == j && nchild[j] == 1
        extra = chain ? (1 + colcount[j]) - colcount[j - 1] : typemax(Int)
        cont = chain && extra <= relax
        relaxed |= cont && extra > 0
        if !cont
            nsuper += 1
            push!(sstart, j)
        end
        super_of[j] = nsuper
    end
    push!(sstart, n + 1)
    # Expand interior column structures to the dense front layout only when a
    # relaxed merge actually widened a supernode (fundamental merges already nest
    # exactly, so this would be a no-op there — skip it to stay bit-identical).
    if relaxed
        @inbounds for sk in 1:nsuper
            c1 = sstart[sk]
            c2 = sstart[sk + 1] - 1
            c2 > c1 || continue
            upd = colstruct[c2]
            nu = length(upd)
            for c in c1:(c2 - 1)
                col = Vector{Int}(undef, (c2 - c) + nu)
                t = 0
                for r in (c + 1):c2
                    t += 1
                    col[t] = r
                end
                copyto!(col, t + 1, upd, 1, nu)   # rows c+1..c2 < c2 < all of upd
                colstruct[c] = col
            end
        end
    end
    childsupers = [Int[] for _ in 1:nsuper]
    @inbounds for sk in 1:nsuper
        lastcol = sstart[sk + 1] - 1
        p = parent[lastcol]
        p != 0 && push!(childsupers[super_of[p]], sk)
    end
    return sstart, super_of, childsupers
end

"""
    SymbolicMF

Symbolic analysis result for the multifrontal factor: the final fill-reducing
column order `qf`, the postordered elimination tree, per-column L structure, and
the supernode partition.
"""
struct SymbolicMF
    qf::Vector{Int}                  # final column order (AMD ∘ postorder)
    parent::Vector{Int}              # elimination tree in qf numbering
    colstruct::Vector{Vector{Int}}   # col structure of L (qf numbering)
    sstart::Vector{Int}              # supernode column ranges
    childsupers::Vector{Vector{Int}} # child supernodes per supernode
end

function symbolic_mf(A::SparseMatrixCSC; q::AbstractVector{<:Integer} = amd_order_sym(A),
        relax::Integer = 0)
    n = size(A, 2)
    qamd = collect(Int, q)
    # etree + postorder on AMD-reordered pattern, then compose orders
    cp1, ri1 = _sym_csc1(A[qamd, qamd])
    parent1 = _etree(cp1, ri1, n)
    post = _postorder(parent1)
    qf = qamd[post]
    # The postordered pattern is the AMD pattern symmetrically permuted by `post`;
    # derive it from (cp1,ri1) directly instead of slicing/symmetrizing A[qf,qf].
    cpF, riF = _permute_sym_csc1(cp1, ri1, post, n)
    parentF = _etree(cpF, riF, n)
    colstruct = _col_structure(cpF, riF, parentF, n)
    sstart, super_of, childsupers = _supernodes(parentF, colstruct, n; relax = Int(relax))
    return SymbolicMF(qf, parentF, colstruct, sstart, childsupers)
end

# Predicted fill (nnz of L + U, counting both diagonals) for cross-checking the
# symbolic structure against the numeric factor.
function predicted_fill(S::SymbolicMF)
    n = length(S.qf)
    nL = n                            # unit diagonal of L
    @inbounds for j in 1:n
        nL += length(S.colstruct[j])  # strictly-below entries of column j
    end
    return 2 * nL                     # nnz(L)+nnz(U); by symmetry nnz(U)==nnz(L)==nL
end

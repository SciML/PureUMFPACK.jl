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
#
# Threading (opt-in via `threaded=true`): the elimination tree's independent
# subtrees have no data dependence, so each can be factored on its own task with a
# *private* front workspace and a *private* contribution-block arena (the shared
# `Fbuf`/`cbval` assume strict serial postorder and are NOT thread-safe).  The
# upper tree near the root — where fronts are large and parallelism is scarce — is
# done on the original serial path.  To make the threaded result *bit-identical* to
# the serial one (and race-free) the CSC scatter uses precomputed deterministic
# write offsets (`uupdoff`) instead of a shared running cursor, so a supernode's
# entries land in exactly the slots serial postorder would have used regardless of
# which task writes them or when.

# Extend-add a child's contribution block into the parent front `F`.  The CB is an
# m×m dense Schur block stored column-major in arena `cbval` starting at 0-based
# offset `off`; its rows/cols are the global ids in `rows` (= the child's update
# set, aliased from `colstruct`).  Function barrier keeps this O(m²) hot loop fully
# typed for whatever concrete `F` (a Matrix view) is passed.
@inline function _extend_add!(F::AbstractMatrix{Tv}, loc::Vector{Int}, rows,
        cbval::Vector{Tv}, off::Int, m::Int) where {Tv}
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
function _factor_colptrs(sstart::Vector{Int}, colstruct::Vector{Vector{Int}},
        n::Int, ::Type{Ti}) where {Ti}
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

# Precompute the deterministic CSC write offset of each supernode's U-update block.
# In serial postorder, U column J is filled by its contributing supernodes (those
# with J in their update set) in ascending-sk order, each appending its `np` pivot
# rows so the column stays row-sorted (own diagonal last).  We lay that out once:
# `uupdoff[uupdptr[sk] + t - 1]` is the 0-based start within `Urowval`/`Unzval`
# where supernode `sk` writes its rows for its t-th update column.  Reproducing the
# serial layout exactly makes the per-supernode scatter independent of execution
# order, hence race-free under threading and bit-identical to the serial result.
function _uupdate_offsets(sstart::Vector{Int}, colstruct::Vector{Vector{Int}},
        Ucolptr::Vector{Ti}, n::Int) where {Ti}
    nsuper = length(sstart) - 1
    uupdptr = Vector{Int}(undef, nsuper + 1)
    uupdptr[1] = 1
    @inbounds for sk in 1:nsuper
        uupdptr[sk + 1] = uupdptr[sk] + length(colstruct[sstart[sk + 1] - 1])
    end
    uupdoff = Vector{Int}(undef, uupdptr[nsuper + 1] - 1)
    ufill = zeros(Int, n)                       # rows already placed in each U column
    @inbounds for sk in 1:nsuper
        c1 = sstart[sk]
        c2 = sstart[sk + 1] - 1
        np = c2 - c1 + 1
        upd = colstruct[c2]
        nu = length(upd)
        base = uupdptr[sk] - 1
        for t in 1:nu
            J = upd[t]
            uupdoff[base + t] = (Ucolptr[J] - 1) + ufill[J]   # 0-based slot start
            ufill[J] += np
        end
    end
    return uupdptr, uupdoff
end

# Factor one supernode `sk` into the preallocated CSC factor arrays.  All large
# buffers are passed in so the same body serves both the serial path (shared
# Fbuf/cbval) and a parallel task (private Fbuf/cbval).  `loc` is a private
# scatter map (zeroed on entry, restored on exit).  CSC writes use precomputed
# offsets (Lcolptr/Ucolptr/uupdoff) so no shared cursor is touched — the only
# arena mutation is this supernode pushing its own CB, and reads of children CBs,
# which the caller sequences via the elimination-tree dependence.  Returns nothing;
# `cbtop` bookkeeping for the LIFO arena is the caller's (it knows the order).
@inline function _factor_super!(sk::Int, c1::Int, c2::Int,
        loc::Vector{Int}, prow::Vector{Int},
        Vp, Vi, Vx, Vtp, Vti, Vtx, sstart::Vector{Int},
        colstruct::Vector{Vector{Int}}, childsupers::Vector{Vector{Int}},
        Fbuf::Matrix{Tv}, cbval::Vector{Tv}, cboff::Vector{Int}, cbtop_in::Int,
        cbarena::Vector{Vector{Tv}},
        Lcolptr::Vector{Ti}, Lrowval::Vector{Ti}, Lnzval::Vector{Tv},
        Ucolptr::Vector{Ti}, Urowval::Vector{Ti}, Unzval::Vector{Tv},
        uupdptr::Vector{Int}, uupdoff::Vector{Int}, check::Bool) where {Tv, Ti}
    np = c2 - c1 + 1
    upd = colstruct[c2]                   # rows > c2 (sorted), shared by block
    nu = length(upd)
    nf = np + nu

    @inbounds for a in 1:np
        loc[c1 + a - 1] = a
    end
    @inbounds for t in 1:nu
        loc[upd[t]] = np + t
    end

    F = view(Fbuf, 1:nf, 1:nf)            # workspace, zeroed for this front
    fill!(F, zero(Tv))

    # assemble original entries of V belonging to this front
    @inbounds for a in 1:np
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

    # extend-add children contribution blocks (LIFO: consume top-down so the active
    # arena `cbval` pops cleanly back to the level before this supernode's children).
    # A child's CB lives in arena `cbarena[ck]` at 0-based offset `cboff[ck]`; for a
    # child produced on *this* arena (this task / the serial region) we pop it off,
    # but a child factored in a different (subtree) arena is only read — popping its
    # offset would corrupt this arena's top pointer.
    cbtop = cbtop_in
    kids = childsupers[sk]
    @inbounds for ci in length(kids):-1:1
        ck = kids[ci]
        crows = colstruct[sstart[ck + 1] - 1]
        m = length(crows)
        m == 0 && continue                # child produced no CB
        _extend_add!(F, loc, crows, cbarena[ck], cboff[ck], m)
        cbarena[ck] === cbval && (cbtop = cboff[ck])   # pop only same-arena children
    end

    # ---- dense factorization of the pivot block (BLAS-3 getrf) ----
    A11 = view(F, 1:np, 1:np)
    fac = lu!(A11, RowMaximum(); check = false)
    ip = fac.ipiv
    if check && fac.info != 0
        throw(SingularException(c1 + fac.info - 1))
    end

    if nu > 0
        R12 = view(F, 1:np, (np + 1):nf)
        @inbounds for k in 1:np               # apply block row swaps to U-update
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
    @inbounds for a in 1:np
        prow[c1 + a - 1] = orig[a]
    end

    # ---- scatter L (global-row coords) and U (factor coords) into CSC ----
    @inbounds for a in 1:np
        gcol = c1 + a - 1
        # L column gcol is owned solely by this supernode: diagonal (orig[a],1),
        # pivots below, then update rows — contiguous from Lcolptr[gcol].
        lp = Lcolptr[gcol]
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
        # U own upper-triangular part is the *last* (gcol-c1+1) slots of column gcol
        # (own rows c1..gcol are the largest, so they trail any descendant fill).
        up = Ucolptr[gcol + 1] - (gcol - c1 + 1)
        for b in 1:a
            Urowval[up] = c1 + b - 1
            Unzval[up] = A11[b, a]
            up += 1
        end
    end
    # U update part: each update column upd[t] receives this supernode's pivot rows
    # c1..c2 at its precomputed slot, values F[a, np+t].  No shared cursor.
    ubase = uupdptr[sk] - 1
    @inbounds for t in 1:nu
        J = upd[t]
        up = uupdoff[ubase + t]
        for a in 1:np
            up += 1
            Urowval[up] = c1 + a - 1
            Unzval[up] = F[a, np + t]
        end
    end

    # push this supernode's Schur complement (F22) onto its arena for the parent.
    if nu > 0
        _ensure_cap!(cbval, cbtop + nu * nu)
        cboff[sk] = cbtop
        cbarena[sk] = cbval
        base = cbtop
        @inbounds for s in 1:nu
            bs = base + (s - 1) * nu
            for r in 1:nu
                cbval[bs + r] = F[np + r, np + s]
            end
        end
        cbtop += nu * nu
    end

    @inbounds for a in 1:np
        loc[c1 + a - 1] = 0
    end
    @inbounds for t in 1:nu
        loc[upd[t]] = 0
    end
    return cbtop
end

"""
    multifrontal_lu(A::SparseMatrixCSC; q=nothing, tol=nothing, check=true,
                    threaded=false, parallel_threshold=4096) -> GPLUFactorization

Supernodal multifrontal LU. Returns the same `GPLUFactorization` (`A[p,q]==L*U`)
as [`gplu`](@ref), so it shares the triangular solves. `q` defaults to the AMD +
postorder ordering from [`symbolic_mf`](@ref).

`threaded=true` factors independent subtrees of the elimination tree concurrently
on Julia threads (each with a private front workspace and contribution-block
arena), falling back to the serial path for the upper tree near the root.  It is
only engaged when `Threads.nthreads() > 1` and `size(A,2) ≥ parallel_threshold`,
and produces output bit-identical to the serial path.  Serial is the default.
"""
function multifrontal_lu(A::SparseMatrixCSC{Tv, Ti}; q = nothing, tol = nothing,
        check::Bool = true, threaded::Bool = false,
        parallel_threshold::Integer = 4096) where {Tv, Ti <: Integer}
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

    prow = collect(1:n)                       # elim step -> global V-row (pivot)

    # largest front (for the serial workspace sizing)
    maxnf = 0
    @inbounds for sk in 1:nsuper
        nfk = (sstart[sk + 1] - sstart[sk]) + length(colstruct[sstart[sk + 1] - 1])
        nfk > maxnf && (maxnf = nfk)
    end

    # Preallocated CSC factor storage; numerics scattered straight to these offsets.
    Lcolptr, Ucolptr = _factor_colptrs(sstart, colstruct, n, Ti)
    Lrowval = Vector{Ti}(undef, Lcolptr[n + 1] - 1)   # L rows in GLOBAL-V coords here
    Lnzval = Vector{Tv}(undef, Lcolptr[n + 1] - 1)
    Urowval = Vector{Ti}(undef, Ucolptr[n + 1] - 1)   # U rows in factor/elim coords
    Unzval = Vector{Tv}(undef, Ucolptr[n + 1] - 1)
    uupdptr, uupdoff = _uupdate_offsets(sstart, colstruct, Ucolptr, n)

    # Per-supernode CB location: which arena holds it, and the 0-based offset there.
    # `cbarena` is fully initialized to one shared empty vector (NOT left `#undef`):
    # the threaded path writes its elements concurrently, and a `Vector{Vector}` with
    # `#undef` slots scanned by the GC while those slots are being filled concurrently
    # corrupts the heap.  Every real slot is overwritten before it is read.
    cboff = zeros(Int, nsuper)
    empty_arena = Tv[]
    cbarena = fill(empty_arena, nsuper)

    use_threads = threaded && Threads.nthreads() > 1 && n >= parallel_threshold

    if use_threads
        _factor_threaded!(sstart, colstruct, childsupers, nsuper, n, maxnf,
            prow, Vp, Vi, Vx, Vtp, Vti, Vtx,
            cboff, cbarena, Lcolptr, Lrowval, Lnzval, Ucolptr, Urowval, Unzval,
            uupdptr, uupdoff, check)
    else
        Fbuf = Matrix{Tv}(undef, maxnf, maxnf)
        cbval = Vector{Tv}(undef, max(64, maxnf * maxnf))
        loc = zeros(Int, n)
        cbtop = 0
        @inbounds for sk in 1:nsuper
            cbtop = _factor_super!(sk, sstart[sk], sstart[sk + 1] - 1,
                loc, prow, Vp, Vi, Vx, Vtp, Vti, Vtx, sstart,
                colstruct, childsupers, Fbuf, cbval, cboff, cbtop, cbarena,
                Lcolptr, Lrowval, Lnzval, Ucolptr, Urowval, Unzval,
                uupdptr, uupdoff, check)
        end
    end

    # Relabel L's row indices from global-V coords to factor order in one O(nnz)
    # pass.  rowfac[g] = factor row of global V-row g (= invperm(prow)).
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

# Threaded driver.  Choose a frontier of subtree roots whose subtrees are factored
# concurrently (each on a private front workspace + private CB arena), then factor
# the remaining upper supernodes serially on the shared workspace.  Independent
# subtrees touch disjoint pivot columns and the CSC scatter uses precomputed
# offsets, so the only cross-task interaction is a parent reading a child CB —
# never within one parallel level, and the serial upper tree runs after the join.
function _factor_threaded!(sstart::Vector{Int}, colstruct::Vector{Vector{Int}},
        childsupers::Vector{Vector{Int}}, nsuper::Int, n::Int, maxnf::Int,
        prow::Vector{Int}, Vp, Vi, Vx, Vtp, Vti, Vtx,
        cboff::Vector{Int}, cbarena::Vector{Vector{Tv}},
        Lcolptr::Vector{Ti}, Lrowval::Vector{Ti}, Lnzval::Vector{Tv},
        Ucolptr::Vector{Ti}, Urowval::Vector{Ti}, Unzval::Vector{Tv},
        uupdptr::Vector{Int}, uupdoff::Vector{Int}, check::Bool) where {Tv, Ti}
    # parent supernode of each supernode (0 = root): parent of sk is the owner of
    # the column that sk's last column elim-points to, recovered from childsupers.
    parent = zeros(Int, nsuper)
    @inbounds for sk in 1:nsuper
        for ck in childsupers[sk]
            parent[ck] = sk
        end
    end

    # Pick subtree roots as an antichain (frontier) of the elimination forest:
    # each chosen subtree is factored by one task, so the chosen roots must be
    # mutually non-ancestral (no root may be a descendant of another) — otherwise
    # two tasks would factor the overlapping supernodes concurrently and race on
    # the shared `prow`/CSC arrays.  Build the frontier top-down: start at the
    # forest roots; a node becomes a subtree root when its subtree is small enough
    # to be one task's chunk, otherwise it stays in the serial upper tree and we
    # descend into its children.  `cap` targets a handful of subtrees per thread.
    subsize = ones(Int, nsuper)               # supernodes in each subtree (incl self)
    @inbounds for sk in 1:nsuper              # postorder: children precede parents
        p = parent[sk]
        p != 0 && (subsize[p] += subsize[sk])
    end
    nthr = Threads.nthreads()
    cap = max(1, nsuper ÷ (4 * nthr))         # cut subtrees no larger than this

    isroot = falses(nsuper)                   # chosen subtree roots (the frontier)
    incut = falses(nsuper)                    # supernode belongs to some chosen subtree
    stack = Int[]
    @inbounds for sk in 1:nsuper              # seed with forest roots
        parent[sk] == 0 && push!(stack, sk)
    end
    @inbounds while !isempty(stack)
        sk = pop!(stack)
        # A leaf, or a subtree small enough for one task, becomes a subtree root.
        # Otherwise keep sk in the serial upper tree and descend to split further.
        if isempty(childsupers[sk]) || subsize[sk] <= cap
            isroot[sk] = true
            _mark_subtree!(incut, childsupers, sk)
        else
            for ck in childsupers[sk]
                push!(stack, ck)
            end
        end
    end

    roots = findall(isroot)
    # order subtree roots largest-first for better load balance
    sort!(roots; by = sk -> subsize[sk], rev = true)

    # Factor each chosen subtree concurrently, each with a private workspace + arena.
    # NOTE: BLAS thread count is deliberately left untouched here.  Toggling
    # `BLAS.set_num_threads` from the main task while the spawned worker tasks are
    # mid-`getrf`/`gemm` reallocates OpenBLAS's per-thread internal buffers and
    # corrupts those in-flight calls (a nondeterministic numeric race).  Measuring
    # tree parallelism wants BLAS=1, but that is the caller's responsibility (set it
    # before calling); it must not be flipped underneath running BLAS.
    @sync begin
        for r in roots
            Threads.@spawn _factor_subtree!(r, sstart, colstruct, childsupers,
                subsize[r], prow, Vp, Vi, Vx, Vtp, Vti, Vtx,
                cboff, cbarena, Lcolptr, Lrowval, Lnzval, Ucolptr, Urowval,
                Unzval, uupdptr, uupdoff, check, n)
        end
    end

    # serial upper tree: every supernode not inside a chosen subtree, in postorder.
    Fbuf = Matrix{Tv}(undef, maxnf, maxnf)
    cbval = Vector{Tv}(undef, max(64, maxnf * maxnf))
    loc = zeros(Int, n)
    cbtop = 0
    @inbounds for sk in 1:nsuper
        incut[sk] && continue
        cbtop = _factor_super!(sk, sstart[sk], sstart[sk + 1] - 1,
            loc, prow, Vp, Vi, Vx, Vtp, Vti, Vtx, sstart,
            colstruct, childsupers, Fbuf, cbval, cboff, cbtop, cbarena,
            Lcolptr, Lrowval, Lnzval, Ucolptr, Urowval, Unzval,
            uupdptr, uupdoff, check)
    end
    return nothing
end

# Mark every supernode in the subtree rooted at `r` (inclusive) in `incut`.
function _mark_subtree!(incut::BitVector, childsupers::Vector{Vector{Int}}, r::Int)
    stack = Int[r]
    @inbounds while !isempty(stack)
        sk = pop!(stack)
        incut[sk] = true
        for ck in childsupers[sk]
            push!(stack, ck)
        end
    end
    return nothing
end

# Factor the subtree rooted at `r` on a private front workspace + private CB arena,
# in postorder (descendants before ancestors), exactly as the serial kernel would
# for that chunk.  The root's CB is left in this private arena for the serial upper
# tree to absorb (recorded via cboff/cbarena).  Sized to the subtree's own largest
# front to keep per-task memory proportional to its work.
function _factor_subtree!(r::Int, sstart::Vector{Int}, colstruct::Vector{Vector{Int}},
        childsupers::Vector{Vector{Int}}, subn::Int,
        prow::Vector{Int}, Vp, Vi, Vx, Vtp, Vti, Vtx,
        cboff::Vector{Int}, cbarena::Vector{Vector{Tv}},
        Lcolptr::Vector{Ti}, Lrowval::Vector{Ti}, Lnzval::Vector{Tv},
        Ucolptr::Vector{Ti}, Urowval::Vector{Ti}, Unzval::Vector{Tv},
        uupdptr::Vector{Int}, uupdoff::Vector{Int}, check::Bool, n::Int) where {Tv, Ti}
    # supernodes of this subtree in postorder (children before parent)
    order = Vector{Int}(undef, subn)
    _subtree_postorder!(order, childsupers, r)

    # largest front within this subtree, and the total CB storage it can ever hold.
    # Pre-size the arena to that upper bound so the per-task LIFO never has to `resize!`
    # mid-loop; the LIFO peak is ≤ this sum, so one allocation suffices.
    lnf = 0
    cbcap = 0
    @inbounds for sk in order
        nu = length(colstruct[sstart[sk + 1] - 1])
        nfk = (sstart[sk + 1] - sstart[sk]) + nu
        nfk > lnf && (lnf = nfk)
        cbcap += nu * nu
    end
    Fbuf = Matrix{Tv}(undef, lnf, lnf)
    cbval = Vector{Tv}(undef, max(64, cbcap))
    loc = zeros(Int, n)
    cbtop = 0
    @inbounds for sk in order
        cbtop = _factor_super!(sk, sstart[sk], sstart[sk + 1] - 1,
            loc, prow, Vp, Vi, Vx, Vtp, Vti, Vtx, sstart,
            colstruct, childsupers, Fbuf, cbval, cboff, cbtop, cbarena,
            Lcolptr, Lrowval, Lnzval, Ucolptr, Urowval, Unzval,
            uupdptr, uupdoff, check)
    end
    return nothing
end

# Fill `order` with the postorder of the subtree rooted at `r` (children, in the
# same ascending order the serial kernel uses, before their parent).  Iterative to
# avoid deep recursion on tall trees.
function _subtree_postorder!(order::Vector{Int}, childsupers::Vector{Vector{Int}}, r::Int)
    k = 0
    stack = Tuple{Int, Int}[(r, 0)]           # (supernode, next-child index)
    @inbounds while !isempty(stack)
        sk, ci = pop!(stack)
        kids = childsupers[sk]
        if ci < length(kids)
            push!(stack, (sk, ci + 1))
            push!(stack, (kids[ci + 1], 0))
        else
            k += 1
            order[k] = sk
        end
    end
    return nothing
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

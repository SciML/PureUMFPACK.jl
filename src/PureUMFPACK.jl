# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2005-2023 Timothy A. Davis (UMFPACK, SuiteSparse) -- GPL-2.0-or-later
# SPDX-FileCopyrightText: 2006 Timothy A. Davis (CSparse, SuiteSparse) -- LGPL-2.1-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# PureUMFPACK.jl is a Julia port of SuiteSparse UMFPACK and CSparse, distributed
# under the GNU GPL v2 or later; see the LICENSE and NOTICE files. SuiteSparse is
# by Timothy A. Davis -- http://www.suitesparse.com (used by permission).

module PureUMFPACK

using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nnz, sparse
using LinearAlgebra: SingularException, RowMaximum, UpperTriangular,
    UnitLowerTriangular, lu!, ldiv!, rdiv!, mul!

export gplu, GPLUFactorization, amd_order_sym, colamd_order, row_scaling,
    splu, PureLU, SCALE_NONE, SCALE_SUM, SCALE_MAX, multifrontal_lu

# `solve` is documented API but stays unexported because SciMLBase exports the same
# name, so `using PureUMFPACK` alongside any SciML package would shadow `solve` and
# break `solve(prob, alg)`. Reach it as `PureUMFPACK.solve`, or use `\`.
# `public` requires Julia >= 1.11; on the 1.10 LTS this is a no-op.
@static if VERSION >= v"1.11"
    include_string(@__MODULE__, "public solve")
end

@inline _colptr(A::SparseMatrixCSC) = getfield(A, :colptr)

include("gplu.jl")
include("solve.jl")
include("scaling.jl")
include("amd.jl")
include("symbolic.jl")
include("multifrontal.jl")
include("interface.jl")

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        A = sparse([2.0 1.0; 1.0 2.0])
        b = [1.0, 0.0]
        for F in (gplu(A), splu(A), multifrontal_lu(A))
            size(F)
            F \ b
        end
        solve(splu(A), b; refine = 1)
    end
end

end # module

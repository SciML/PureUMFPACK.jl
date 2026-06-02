# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-FileCopyrightText: 2005-2023 Timothy A. Davis (UMFPACK, SuiteSparse) -- GPL-2.0-or-later
# SPDX-FileCopyrightText: 2006 Timothy A. Davis (CSparse, SuiteSparse) -- LGPL-2.1-or-later
# SPDX-License-Identifier: GPL-2.0-or-later
#
# PureUMFPACK.jl is a Julia port of SuiteSparse UMFPACK and CSparse, distributed
# under the GNU GPL v2 or later; see the LICENSE and NOTICE files. SuiteSparse is
# by Timothy A. Davis -- http://www.suitesparse.com (used by permission).

module PureUMFPACK

using SparseArrays
using SparseArrays: getcolptr, rowvals, nonzeros
using LinearAlgebra
using LinearAlgebra: SingularException, RowMaximum, UpperTriangular,
    UnitLowerTriangular, lu!, ldiv!, rdiv!, mul!

export gplu, GPLUFactorization, solve, amd_order_sym, colamd_order, row_scaling,
    splu, PureLU, SCALE_NONE, SCALE_SUM, SCALE_MAX, multifrontal_lu

include("gplu.jl")
include("solve.jl")
include("scaling.jl")
include("amd.jl")
include("symbolic.jl")
include("multifrontal.jl")
include("interface.jl")

end # module

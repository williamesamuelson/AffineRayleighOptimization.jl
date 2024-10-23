module AffineRayleighOptimization
using LinearAlgebra
import SciMLBase: solve, init, solve!
using TestItems
using LinearMaps
using KrylovKit

# export QuadraticForm
export ConstrainedProblem, ConstrainedRayleighQuotientProblem, Span
export QF_BACKSLASH, QF_LINEARSOLVE

export solve

include("quadratic_form.jl")
include("rayleigh_quotient.jl")

end

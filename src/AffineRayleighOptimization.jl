module AffineRayleighOptimization
using LinearAlgebra
import SciMLBase: solve, init, solve!
using TestItems
using LinearMaps
using KrylovKit

export QuadraticProblem, RayleighProblem, Span
export QF_BACKSLASH, QF_LINEARSOLVE

export solve

include("quadratic_form.jl")
include("span_problem.jl")

end

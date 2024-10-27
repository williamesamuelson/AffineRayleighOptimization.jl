module AffineRayleighOptimization
using LinearAlgebra
import SciMLBase: solve, init, solve!
using TestItems
using LinearMaps
using KrylovKit

export QuadraticProblem, RayleighProblem
export QF_BACKSLASH, QF_LINEARSOLVE

export solve

include("quadratic_form.jl")
include("rayleigh_problem.jl")

end

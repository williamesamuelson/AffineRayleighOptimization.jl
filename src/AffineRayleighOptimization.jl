module AffineRayleighOptimization
using LinearAlgebra
using LinearSolve
import SciMLBase: solve
using TestItems

export RayleighQuotient, QuadraticForm
export ConstrainedRayleighQuotientProblem, ConstrainedQuadraticFormProblem
export solve

include("quadratic_form.jl")
include("rayleigh_quotient.jl")

end

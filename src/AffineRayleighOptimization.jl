module AffineRayleighOptimization
using LinearAlgebra
using LinearSolve
import SciMLBase: solve
using TestItems

include("quadratic_form.jl")
include("rayleigh_quotient.jl")

end

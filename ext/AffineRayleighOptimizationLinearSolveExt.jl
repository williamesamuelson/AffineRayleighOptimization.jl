module AffineRayleighOptimizationLinearSolveExt

using AffineRayleighOptimization
using LinearAlgebra
using LinearSolve


struct ConstrainedQuadraticFormLinearSolveSolver{LP,TM}
    linearprob::LP
    transform_matrix::TM
end

function AffineRayleighOptimization.init(prob::ConstrainedQuadraticFormProblem, alg::A, args...; kwargs...) where {A<:Union{<:SciMLBase.AbstractLinearAlgorithm,<:QF_LINEARSOLVE}}
    inv_penalty_mat = inv(prob.Q.quadratic_form)
    original_lhs_mat = prob.C
    tm = inv_penalty_mat * original_lhs_mat'
    new_lhs = Hermitian(original_lhs_mat * tm)
    linearprob = init(LinearProblem(new_lhs, prob.b; kwargs...), alg, args...; kwargs...)
    ConstrainedQuadraticFormLinearSolveSolver(linearprob, tm)
end

AffineRayleighOptimization.init(prob::LinearProblem, alg::QF_LINEARSOLVE{A}; kwargs...) where {A<:SciMLBase.AbstractLinearAlgorithm} = init(prob, alg.alg; kwargs...)


function AffineRayleighOptimization.solve!(prob::ConstrainedQuadraticFormLinearSolveSolver)
    intermediate_sol = solve!(prob.linearprob)
    return prob.transform_matrix * intermediate_sol
end

end
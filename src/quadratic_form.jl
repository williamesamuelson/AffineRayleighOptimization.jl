"""
    QuadraticProblem(Q, C, b)

A constrained quadratic form problem of the form
    minimize `dot(x,Qx)`
    subject to `Cx = b`.
"""
struct QuadraticProblem{Q,C,B}
    Q::Q
    C::C
    b::B
    function QuadraticProblem(_Q, C, b)
        Q = Hermitian(_Q)
        return new{typeof(Q),typeof(C),typeof(b)}(Q, C, b)
    end
end


abstract type QF_ALG end
struct QUADRATIC_BACKSLASH <: QF_ALG end

struct ConstrainedQuadraticFormBackslashSolver{L,B,TM}
    lhs::L
    b::B
    transform_matrix::TM
end
default_linear_alg(::QuadraticProblem) = QUADRATIC_BACKSLASH()
init(prob::QuadraticProblem) = init(prob, default_linear_alg(prob))
# https://dept.math.lsa.umich.edu/~speyer/417/Minimization.pdf
function init(prob::QuadraticProblem, alg::QUADRATIC_BACKSLASH)
    inv_penalty_mat = inv(prob.Q)
    original_lhs_mat = prob.C
    tm = inv_penalty_mat * original_lhs_mat'
    new_lhs = Hermitian(original_lhs_mat * tm)
    ConstrainedQuadraticFormBackslashSolver(new_lhs, prob.b, tm)
end

function solve!(prob::ConstrainedQuadraticFormBackslashSolver)
    intermediate_sol = prob.lhs \ prob.b
    return prob.transform_matrix * intermediate_sol
end


@testitem "QuadraticProblem" begin
    using LinearAlgebra, Random
    using AffineRayleighOptimization: QUADRATIC_LINEARSOLVE
    Random.seed!(1234)
    Q = Diagonal(1:10)
    C = I
    b = rand(10)
    prob = QuadraticProblem(Q, C, b)
    sol = solve(prob)
    @test sol ≈ b
    using LinearSolve
    sol = solve(prob, QUADRATIC_LINEARSOLVE()) #Will use the default solver from LinearSolve
    @test sol ≈ b
    sol = solve(prob, KrylovJL_MINRES()) #Will use the KrylovJL_MINRES solver from LinearSolve
    @test sol ≈ b
    sol = solve(prob, QUADRATIC_LINEARSOLVE(KrylovJL_MINRES())) #Will use the KrylovJL_MINRES solver from LinearSolve
    @test sol ≈ b
    sol = solve(prob, KrylovJL_MINRES(); u0=rand(10)) # Set initial guess
    @test sol ≈ b
end


@kwdef struct QUADRATIC_LINEARSOLVE{A}
    alg::A = nothing
end

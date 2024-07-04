"""
    QuadraticForm(quadratic_form)

Quadratic form representation.
"""
struct QuadraticForm{Q}
    quadratic_form::Q
    function QuadraticForm(q::AbstractMatrix)
        Hq = Hermitian(q)
        new{typeof(Hq)}(Hq)
    end
end
QuadraticForm(qf::QuadraticForm) = qf
(rq::QuadraticForm)(x) = dot(x, rq.quadratic_form, x)

@testitem "QuadraticForm" begin
    using LinearAlgebra
    N = 10
    x = rand(N)
    Q = I(N)
    qf = QuadraticForm(Q)
    @test qf(x) ≈ dot(x, x)
end

"""
    ConstrainedQuadraticFormProblem(Q, C, b)

A constrained quadratic form problem of the form
    minimize `dot(x,Qx)`
    subject to `Cx = b`.
"""
struct ConstrainedQuadraticFormProblem{Q,C,B}
    Q::QuadraticForm{Q}
    C::C
    b::B
end
function ConstrainedQuadraticFormProblem(Q, C, b)
    return ConstrainedQuadraticFormProblem(QuadraticForm(Q), C, b)
end

abstract type QF_ALG end
struct QF_BACKSLASH <: QF_ALG end

struct ConstrainedQuadraticFormBackslashSolver{L,B,TM}
    lhs::L
    b::B
    transform_matrix::TM
end
default_linear_alg(prob::ConstrainedQuadraticFormProblem) = QF_BACKSLASH()

# https://dept.math.lsa.umich.edu/~speyer/417/Minimization.pdf
function init(prob::ConstrainedQuadraticFormProblem, alg::QF_BACKSLASH=default_linear_alg(prob))
    inv_penalty_mat = inv(prob.Q.quadratic_form)
    original_lhs_mat = prob.C
    tm = inv_penalty_mat * original_lhs_mat'
    new_lhs = Hermitian(original_lhs_mat * tm)
    ConstrainedQuadraticFormBackslashSolver(new_lhs, prob.b, tm)
end

function solve!(prob::ConstrainedQuadraticFormBackslashSolver)
    intermediate_sol = prob.lhs \ prob.b
    return prob.transform_matrix * intermediate_sol
end


@testitem "QuadraticFormProblem" begin
    using LinearAlgebra, Random
    Random.seed!(1234)
    Q = Diagonal(1:10)
    C = I
    b = rand(10)
    prob = ConstrainedQuadraticFormProblem(Q, C, b)
    sol = solve(prob)
    @test sol ≈ b
    using LinearSolve
    sol = solve(prob, QF_LINEARSOLVE()) #Will use the default solver from LinearSolve
    @test sol ≈ b
    sol = solve(prob, KrylovJL_MINRES()) #Will use the KrylovJL_MINRES solver from LinearSolve
    @test sol ≈ b
    sol = solve(prob, QF_LINEARSOLVE(KrylovJL_MINRES())) #Will use the KrylovJL_MINRES solver from LinearSolve
    @test sol ≈ b
    sol = solve(prob, KrylovJL_MINRES(); u0=rand(10)) # Set initial guess
    @test sol ≈ b
end


@kwdef struct QF_LINEARSOLVE{A}
    alg::A = nothing
end

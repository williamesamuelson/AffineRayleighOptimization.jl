"""
    QuadraticForm(quadratic_form)

Create a quadratic form representation.
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

# https://dept.math.lsa.umich.edu/~speyer/417/Minimization.pdf
function solve(prob::ConstrainedQuadraticFormProblem, alg=KrylovJL_MINRES())
    inv_penalty_mat = inv(prob.Q.quadratic_form)
    original_lhs_mat = prob.C
    new_lhs = Hermitian(original_lhs_mat * inv_penalty_mat * original_lhs_mat')
    intermediate_sol = solve(LinearProblem(new_lhs, prob.b), alg).u
    return inv_penalty_mat * original_lhs_mat' * intermediate_sol
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
end

"""
    RayleighQuotient(q)

Rayleigh quotient representation.
"""
struct RayleighQuotient{Q}
    quadratic_form::Q
    function RayleighQuotient(q::AbstractMatrix)
        Hq = Hermitian(q)
        new{typeof(Hq)}(Hq)
    end
end
RayleighQuotient(rq::RayleighQuotient) = rq
rq_type(::RayleighQuotient{Q}) where {Q} = Q
(rq::RayleighQuotient)(x) = dot(x, rq.quadratic_form, x) / dot(x, x)

@testitem "RayleighQuotient" begin
    using LinearAlgebra
    N = 10
    x = rand(N)
    Q = I(N)
    rq = RayleighQuotient(Q)
    @test rq(x) ≈ 1
end

"""
    ConstrainedRayleighQuotientProblem(Q, C, b)

A constrained quadratic form problem of the form
    minimize `dot(x,Qx)/dot(x,x)`
    subject to `Cx = b`.

b can also be given as a span, using the Span struct.
"""
struct ConstrainedRayleighQuotientProblem{Q,Cmat,B}
    Q::RayleighQuotient{Q}
    C::Cmat
    b::B
    function ConstrainedRayleighQuotientProblem(q::Q, C::Cmat, b::B) where {Q,Cmat,B}
        rq = RayleighQuotient(q)
        return new{rq_type(rq),Cmat,B}(rq, C, b)
    end
    function ConstrainedRayleighQuotientProblem(q::RayleighQuotient{Q}, C::Cmat, b::B) where {Q,Cmat,B}
        return new{Q,Cmat,B}(RayleighQuotient(q), C, b)
    end
end

"""
    Span(vecs::AbstractMatrix)

Representation of a span, which can be given as `b` in the ConstrainedRayleighQuotientProblem.
The vectors are provided as the columns in vecs.
"""
struct Span{V<:AbstractMatrix}
    vecs::V
end

_get_b(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:AbstractVector}) where {Q, Cmat} = prob.b
_get_b(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}) where {Q, Cmat} = prob.b.vecs

abstract type RQ_ALG end
struct RQ_CHOL <: RQ_ALG end
struct RQ_GENEIG <: RQ_ALG end
const DEFAULT_SELECTVEC_EPS = 1e-15
@kwdef struct RQ_EIG{T} <: RQ_ALG
    eps::T = DEFAULT_SELECTVEC_EPS
end
@kwdef struct RQ_HOMO{T} <: RQ_ALG
    eps::T = DEFAULT_SELECTVEC_EPS
end
@kwdef struct RQ_SPARSE{T} <: RQ_ALG
    eps::T = DEFAULT_SELECTVEC_EPS
end

solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:AbstractVector}) where {Q,Cmat} = solve(prob, RQ_GENEIG())
solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}) where {Q,Cmat} = solve(prob, RQ_EIG())
const SPAN_INCOMPATIBLE_ALGS = Union{RQ_GENEIG, RQ_CHOL, RQ_SPARSE, RQ_HOMO}
function solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}, alg::SPAN_INCOMPATIBLE_ALGS) where {Q,Cmat}
    error("$alg is incompatible with b a Span. Use RQ_EIG() instead.")
end
function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_GENEIG)
    augC = hcat(prob.C, -prob.b)
    N = nullspace(augC)
    Ntrunc = N[1:end-1, :]
    Nv = N[end, :]
    new_quadratic_form = Hermitian(Ntrunc' * prob.Q.quadratic_form * Ntrunc)
    rhs_gen_eigen = Hermitian(Ntrunc' * Ntrunc)
    eigsol = eigen(new_quadratic_form, rhs_gen_eigen).vectors[:, 1] # smallest eigenvalue eigenvector maximizes
    unnormalized_sol = Ntrunc * eigsol
    return unnormalized_sol / dot(Nv, eigsol)
end

function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_CHOL)
    augC = hcat(prob.C, -prob.b)
    N = nullspace(augC)
    Ntrunc = N[1:end-1, :]
    Nv = N[end, :]
    upper = cholesky(Ntrunc' * Ntrunc).U
    inv_upper = inv(upper)
    new_quadratic_form = Hermitian(inv_upper' * Ntrunc' * prob.Q.quadratic_form * Ntrunc * inv_upper)
    eigsol = eigen(new_quadratic_form, 1:1).vectors[:, 1]
    unnormalized_sol = Ntrunc * inv_upper * eigsol
    return unnormalized_sol / dot(Nv, inv_upper * eigsol)
end


#https://www.cis.upenn.edu/~jshi/papers/supplement_nips2006.pdf
function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_EIG)
    b = _get_b(prob) # fetch b matrix/vector if prob.b is span/vector
    C = prob.C
    N = size(b, 1)
    Kb = I - b * b' / dot(b, b)
    _, inds_keep = _inds_to_remove_and_keep(b, N)
    augC = (Kb * C)[inds_keep, :] # instead of J * Kb * C where J = I(N)[inds_keep, :]
    return _solve_homo_prob(augC, prob, alg)
end

function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_SPARSE)
    b = _get_b(prob)
    C = prob.C
    inds_remove, inds_keep = _inds_to_remove_and_keep(b, length(b))
    bk = first(b[inds_remove]) # only one element, no support for b Span
    Ck = C[inds_remove, :]
    augC = (C - (1/bk) * b * Ck)[inds_keep, :] # replace mult by J with slicing, see above
    return _solve_homo_prob(augC, prob, alg)
end

function _inds_to_remove_and_keep(b, N)
    inds_remove = partialsortperm(eachrow(b), 1:size(b, 2); by=norm, rev=true)
    inds_keep = map(!in(inds_remove), 1:N)
    return inds_remove, inds_keep
end

function _solve_homo_prob(augC, prob::ConstrainedRayleighQuotientProblem, alg)
    b = _get_b(prob)
    C = prob.C
    homo_prob = ConstrainedRayleighQuotientProblem(prob.Q, augC, zero(b))
    sol = solve(homo_prob, RQ_HOMO(alg.eps))
    t = dot(b, C, sol) / dot(b, b)
    return sol / t
end

function _solve_homo_prob(augC, prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span} , alg) where {Q,Cmat}
    b = _get_b(prob)[:, 1] # for b Span, normalization is not specified
    homo_prob = ConstrainedRayleighQuotientProblem(prob.Q, augC, zero(b))
    return solve(homo_prob, RQ_HOMO(alg.eps)) # returns a view, is that ok?
end

function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_HOMO)
    @assert iszero(prob.b) "b must be zero"
    C = prob.C
    P = I - C' * (factorize(Hermitian(C * C')) \ C)
    eig = eigen(Hermitian(P' * prob.Q.quadratic_form * P))
    _select_vectors(eig.vectors, P, alg.eps)
end

function _select_vectors(eigvecs, P, eps=DEFAULT_SELECTVEC_EPS)
    vecs = eachcol(eigvecs)
    v = similar(first(vecs))
    f = x -> any(>(eps) ∘ abs, mul!(v, P, x))
    i1 = findfirst(f, vecs)
    return vecs[i1]
end

## Tests
@testitem "RayleighQuotientProblem" begin
    using LinearAlgebra, Random
    import AffineRayleighOptimization: RQ_GENEIG, RQ_CHOL, RQ_EIG, RQ_SPARSE
    function test_prob_known_sol(prob, known_sol, solvers=[RQ_GENEIG(), RQ_CHOL(), RQ_EIG(), RQ_SPARSE()])
        for solver in solvers
            sol = solve(prob, solver)
            @test sol ≈ known_sol
        end
    end
    function test_prob(prob, solvers=[RQ_GENEIG(), RQ_CHOL(), RQ_EIG(), RQ_SPARSE()])
        sols = []
        for solver in solvers
            sol = solve(prob, solver)
            push!(sols, sol)
        end
        @test all(x->x ≈ first(sols), sols[2:end])
    end
    Random.seed!(1234)
    #prob 1 (sol=b)
    Q = Diagonal(1:10)
    rc = RayleighQuotient(Q)
    @test rc(ones(10)) ≈ sum(Q) / norm(ones(10))^2
    C = I(10)
    b = rand(10)
    prob = ConstrainedRayleighQuotientProblem(rc, C, b)
    test_prob_known_sol(prob, b)
    #prob 2 (span)
    b1 = [0.0, 2.0, zeros(8)...]
    b2 = [0.0, 0.0, 1.0, zeros(7)...]
    bspan = hcat(b1, b2)
    prob = ConstrainedRayleighQuotientProblem(rc, C, Span(bspan))
    test_prob_known_sol(prob, b1/2, [RQ_EIG()])
    bspan = hcat(b2, b1) # change order
    prob = ConstrainedRayleighQuotientProblem(rc, C, Span(bspan))
    test_prob_known_sol(prob, b1/2, [RQ_EIG()])
    #prob 3
    C = ones(1, 10)
    b = [1.0]
    prob = ConstrainedRayleighQuotientProblem(rc, C, b)
    test_prob_known_sol(prob, [1.0, zeros(9)...])
    #prob 4
    n = 20
    k = 10
    rc = RayleighQuotient(Hermitian(rand(n, n)))
    C = rand(k, n)
    b = rand(k)
    prob = ConstrainedRayleighQuotientProblem(rc, C, b)
    test_prob(prob)
end

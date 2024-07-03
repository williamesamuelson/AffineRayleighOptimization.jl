"""
    RayleighQuotient(q)

Create a Rayleigh quotient representation.
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
"""
struct ConstrainedRayleighQuotientProblem{Q,C,B}
    Q::RayleighQuotient{Q}
    C::C
    b::B
    function ConstrainedRayleighQuotientProblem(q::Q, c::C, b::B) where {Q,C,B}
        rq = RayleighQuotient(q)
        return new{rq_type(rq),C,B}(rq, c, b)
    end
    function ConstrainedRayleighQuotientProblem(q::RayleighQuotient{Q}, c::C, b::B) where {Q,C,B}
        return new{Q,C,B}(RayleighQuotient(q), c, b)
    end
end

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

solve(prob::ConstrainedRayleighQuotientProblem) = solve(prob, RQ_GENEIG())
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
    b = prob.b
    C = prob.C
    Kb = I - b * b' / dot(b, b)
    nlargest = partialsortperm(norm.(eachrow(b)), 1:size(b, 2), rev=true)
    N = size(b, 1)
    inds = map(!in(nlargest), 1:N)
    J = I(N)[inds, :]
    augC = J * Kb * C
    homo_prob = ConstrainedRayleighQuotientProblem(prob.Q, augC, zero(b))
    sol = solve(homo_prob, RQ_HOMO(alg.eps))
    z = dot(b, C, sol) / dot(b, b)
    return sol / z
end

function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_HOMO)
    @assert iszero(prob.b) "b must be zero"
    C = prob.C
    P = I - C' * (factorize(Hermitian(C * C')) \ C)
    eig = eigen(Hermitian(P' * prob.Q.quadratic_form * P))
    _select_vectors(eig.vectors, P, prob.b, alg.eps)
end

function _select_vectors(eigvecs, P, b::AbstractMatrix, eps=DEFAULT_SELECTVEC_EPS)
    vecs = eachcol(eigvecs)
    v = similar(first(vecs))
    f = x -> any(>(eps) ∘ abs, mul!(v, P, x))
    i1 = findfirst(f, vecs)
    vecinds = [i1]
    for n in 2:size(b, 2)
        push!(vecinds, findnext(f, vecs, last(vecinds) + 1))
    end
    return reduce(hcat, vecs[vecinds])
end


function _select_vectors(eigvecs, P, b::AbstractVector, eps=DEFAULT_SELECTVEC_EPS)
    vecs = eachcol(eigvecs)
    v = similar(first(vecs))
    f = x -> any(>(eps) ∘ abs, mul!(v, P, x))
    i1 = findfirst(f, vecs)
    return vecs[i1]
end

## Tests
#=@testitem "RayleighQuotientProblem" begin=#
#=    using LinearAlgebra, Random=#
#=    Random.seed!(1234)=#
#=    Q = Diagonal(1:10)=#
#=    rc = RayleighQuotient(Q)=#
#=    @test rc(ones(10)) ≈ sum(Q) / norm(ones(10))^2=#
#=    C = I=#
#=    b = rand(10)=#
#=    prob = WeakMajoranas.ConstrainedRayleighQuotientProblem(rc, C, b)=#
#=    for solver in [WeakMajoranas.RQ_EIG(), WeakMajoranas.RQ_CHOL(), WeakMajoranas.RQ_GENEIG()]=#
#=        sol = solve(prob, solver)=#
#=        @test sol ≈ b=#
#=    end=#
#==#
#=    C = ones(1, 10)=#
#=    b = [1.0]=#
#=    prob = WeakMajoranas.ConstrainedRayleighQuotientProblem(rc, C, b)=#
#=    sol = solve(prob)=#
#=    @test sol ≈ [1.0, zeros(9)...]=#
#=    sol2 = solve(prob, WeakMajoranas.RQ_CHOL())=#
#=    @test sol ≈ sol2=#
#=end=#

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
Span(v::AbstractVector) = Span(hcat(v))
Span(itr...) = Span(reduce(hcat, itr))
@testitem "Span" begin
    b1 = rand(2)
    b2 = rand(2)
    bspan = Span(b1, b2)
    bmatspan = Span(hcat(b1, b2))
    @test bspan.vecs == bmatspan.vecs
    @test Span(b1).vecs[:] == b1
end

_get_b(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:AbstractVector}) where {Q, Cmat} = prob.b
_get_b(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}) where {Q, Cmat} = prob.b.vecs

abstract type RQ_ALG end
struct RQ_CHOL <: RQ_ALG end
struct RQ_GENEIG <: RQ_ALG end
const DEFAULT_SELECTVEC_EPS = 1e-15
@kwdef struct RQ_EIG{T} <: RQ_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end
@kwdef struct RQ_SPARSE{T} <: RQ_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end
@kwdef struct RQ_HOMO{T} <: RQ_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end

solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:AbstractVector}) where {Q,Cmat} = solve(prob, RQ_CHOL())
solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}) where {Q,Cmat} = solve(prob, RQ_EIG())
@testitem "Default solvers" begin
    using Random
    import AffineRayleighOptimization: RQ_EIG, RQ_CHOL
    Random.seed!(1234)
    prob_span = ConstrainedRayleighQuotientProblem(rand(2,2), rand(1, 2), Span(rand(1)))
    prob_vec = ConstrainedRayleighQuotientProblem(rand(2,2), rand(1, 2), rand(1))
    @test solve(prob_span) ≈ solve(prob_span, RQ_EIG()) || solve(prob_span) ≈ -solve(prob_span, RQ_EIG())
    @test solve(prob_vec) ≈ solve(prob_vec, RQ_CHOL())
end

# not so nice, but helps for method ambs
solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}, alg::RQ_GENEIG) where {Q,Cmat} = _span_incomp_error_msg(alg)
solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}, alg::RQ_CHOL) where {Q,Cmat} = _span_incomp_error_msg(alg)
solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}, alg::RQ_SPARSE) where {Q,Cmat} = _span_incomp_error_msg(alg)
solve(prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span}, alg::RQ_HOMO) where {Q,Cmat} = _span_incomp_error_msg(alg)
_span_incomp_error_msg(alg) = error("$alg is incompatible with b a Span. Use RQ_EIG() instead.")

@testitem "Span incompatible" begin
    using Random
    import AffineRayleighOptimization: RQ_GENEIG, RQ_CHOL, RQ_SPARSE, RQ_HOMO
    Random.seed!(1234)
    prob = ConstrainedRayleighQuotientProblem(rand(1,1), rand(1), Span(rand(1)))
    for solver in [RQ_GENEIG(), RQ_CHOL(), RQ_SPARSE(), RQ_HOMO()]
        @test_throws ErrorException solve(prob, solver)
    end
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
function solve(prob::ConstrainedRayleighQuotientProblem, alg::Union{RQ_EIG, RQ_SPARSE})
    C_reduced = _get_C_reduced(prob, alg)
    return _solve_homo_prob(C_reduced, prob, alg)
end

function _get_C_reduced(prob::ConstrainedRayleighQuotientProblem, alg::RQ_EIG)
    b = _get_b(prob) # fetch b matrix/vector if prob.b is span/vector
    C = prob.C
    N = size(b, 1)
    Kb = I - b * b' / dot(b, b)
    _, inds_keep = _inds_to_remove_and_keep(b, N)
    return (Kb * C)[inds_keep, :] # instead of J * Kb * C where J = I(N)[inds_keep, :]
end

function _get_C_reduced(prob::ConstrainedRayleighQuotientProblem, alg::RQ_SPARSE)
    b = _get_b(prob)
    C = prob.C
    inds_remove, inds_keep = _inds_to_remove_and_keep(b, length(b))
    ind_remove = only(inds_remove) # only one element, no support for b Span
    bk = b[ind_remove]
    Ck = transpose(C[ind_remove, :])
    return (C - (1/bk) * b * Ck)[inds_keep, :] # replace mult by J with slicing, see above
end

function _inds_to_remove_and_keep(b, N)
    inds_remove = partialsortperm(eachrow(b), 1:size(b, 2); by=norm, rev=true)
    inds_keep = map(!in(inds_remove), 1:N)
    return inds_remove, inds_keep
end

function _solve_homo_prob(C_reduced, prob::ConstrainedRayleighQuotientProblem, alg)
    b = _get_b(prob)
    C = prob.C
    homo_prob = ConstrainedRayleighQuotientProblem(prob.Q, C_reduced, zero(b))
    sol = solve(homo_prob, RQ_HOMO(alg.krylov_howmany, alg.krylov_kwargs))
    t = dot(b, C, sol) / dot(b, b)
    return sol / t
end

function _solve_homo_prob(augC, prob::ConstrainedRayleighQuotientProblem{Q,Cmat,<:Span} , alg) where {Q,Cmat}
    b = _get_b(prob)[:, 1] # for b Span, normalization is not specified
    homo_prob = ConstrainedRayleighQuotientProblem(prob.Q, augC, zero(b))
    return solve(homo_prob, RQ_HOMO(alg.krylov_howmany, alg.krylov_kwargs))
end

function solve(prob::ConstrainedRayleighQuotientProblem, alg::RQ_HOMO)
    @assert iszero(prob.b) "b must be zero"
    C = prob.C
    P, PQP = _get_P_PQP_homo(C, prob.Q.quadratic_form)
    howmany = 1
    # we need to specify ishermitian to eigsolve. eigsolve doesn't check this if the matrix is not an AbstractMatrix
    _, eigvecs, info = eigsolve(PQP, size(C, 2), howmany, :SR; ishermitian=true, alg.krylov_kwargs...)
    ind = _select_vector_index(eigvecs, P)
    !isnothing(ind) && return eigvecs[ind] # if an ok vector is found, return it, otherwise, look for more
    #=howmany = size(C, 1) + 1 # this is often too large=#
    _, eigvecs, info = eigsolve(PQP, first(eigvecs), alg.krylov_howmany, :SR; ishermitian=true, alg.krylov_kwargs...)
    ind = _select_vector_index(eigvecs, P)
    !isnothing(ind) && return eigvecs[ind]
    error("Solution was not found. Increase alg.krylov_howmany and/or krylovdim in alg.krylov_kwargs to look for more candidates")
end

function _get_P_PQP_homo(C, Q)
    #=F = factorize(Hermitian(C * C')) #sparse cholesky is incompatible with ldiv!=#
    F = lu!(Hermitian(C * C'))
    issymmetric = eltype(C) <: Real # I think this is correct
    P = I - C' * LinearMap(InverseMap(F); ishermitian=true, issymmetric, isposdef=true) * C
    P = LinearMap(P; ishermitian=true)
    PQP = LinearMap(P' * Q * P; ishermitian=true)
    return P, PQP
end

function _select_vector_index(eigvecs, P)
    f = x -> dot(x, P, x) > 1/2
    return findfirst(f, eigvecs)
end

## Tests
@testitem "RayleighQuotientProblem" begin
    using LinearAlgebra, Random, SparseArrays
    import AffineRayleighOptimization: RQ_GENEIG, RQ_CHOL, RQ_EIG, RQ_SPARSE
    all_solvers = [RQ_GENEIG(), RQ_CHOL(), RQ_EIG(), RQ_SPARSE()]
    function test_prob_known_sol(prob, known_sol, solvers=all_solvers)
        for solver in solvers
            sol = solve(prob, solver)
            if prob.b isa Span
                @test (sol ≈ known_sol) || (sol ≈ -known_sol) # sign undetermined if b Span?
            else
                @test sol ≈ known_sol
            end
        end
    end
    function test_prob(prob, solvers=all_solvers)
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
    prob = ConstrainedRayleighQuotientProblem(rc, C, Span(b1, b2))
    test_prob_known_sol(prob, b1/2, [RQ_EIG()])
    prob = ConstrainedRayleighQuotientProblem(rc, C, Span(b2, b1)) # change order
    test_prob_known_sol(prob, b1/2, [RQ_EIG()])
    #prob 3
    C = ones(1, 10)
    b = [1.0]
    prob = ConstrainedRayleighQuotientProblem(rc, C, b)
    test_prob_known_sol(prob, [1.0, zeros(9)...])
    #prob 4 (random matrix)
    n = 20
    k = 10
    rc = RayleighQuotient(Hermitian(rand(n, n)))
    C = rand(k, n)
    b = rand(k)
    prob = ConstrainedRayleighQuotientProblem(rc, C, b)
    test_prob(prob)
    # prob 5 (positive definite matrix)
    M = Hermitian(rand(n, n))
    rc = RayleighQuotient(M'M)
    prob = ConstrainedRayleighQuotientProblem(rc, C, b)
    test_prob(prob)
    # prob 6 (sparse C)
    n = 1000
    k = 300
    Q = Hermitian(sprand(n, n, 1/n))
    rc = RayleighQuotient(Q)
    b = sprand(k, 0.1)
    function generate_sparse_fullrank_C(n, k, sparsity)
        C = zeros(k, n)
        fullrank = false
        while !fullrank
            C = sprand(k, n, sparsity)
            fullrank = rank(C) == k
        end
        return C
    end
    C = generate_sparse_fullrank_C(n, k, 0.01)
    prob_sparse = ConstrainedRayleighQuotientProblem(rc, C, b)
    C_red_sparse = AffineRayleighOptimization._get_C_reduced(prob_sparse, RQ_SPARSE())
    @test C_red_sparse isa SparseMatrixCSC
    sol = solve(prob_sparse, RQ_SPARSE())
    prob_dense = ConstrainedRayleighQuotientProblem(rc, Matrix(C), Vector(b))
    test_prob_known_sol(prob_dense, sol)
    # prob 7 (sparse C with pos def Q)
    M = Hermitian(rand(n, n))
    rc = RayleighQuotient(M'M)
    prob_sparse = ConstrainedRayleighQuotientProblem(rc, C, b)
    # here we need to increase krylov_howmany to find the sol
    @test_throws ErrorException solve(prob_sparse, RQ_SPARSE())
    sol = solve(prob_sparse, RQ_SPARSE(krylov_howmany=10))
    rc_dense = RayleighQuotient(Matrix(rc.quadratic_form))
    prob_dense = ConstrainedRayleighQuotientProblem(rc_dense, Matrix(C), Vector(b))
    test_prob_known_sol(prob_dense, sol, [RQ_CHOL(), RQ_GENEIG()])
end

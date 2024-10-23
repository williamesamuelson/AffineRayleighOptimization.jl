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
Span(s::Span) = s
const SpanProblem = ConstrainedProblem{<:Any,<:Any,<:Span}

@testitem "Span" begin
    b1 = rand(2)
    b2 = rand(2)
    bspan = Span(b1, b2)
    bmatspan = Span(hcat(b1, b2))
    @test bspan.vecs == bmatspan.vecs
    @test Span(b1).vecs[:] == b1
    @test Span(bspan) == bspan
end

_get_b(prob::ConstrainedProblem) = _get_b(prob.b)
_get_b(b) = b
_get_b(b::Span) = b.vecs

ConstrainedRayleighQuotientProblem(Q, c, b) = ConstrainedProblem(Q, c, Span(b))

abstract type SPAN_ALG end
struct SPAN_CHOL <: SPAN_ALG end
struct SPAN_GENEIG <: SPAN_ALG end
const DEFAULT_SELECTVEC_EPS = 1e-15
@kwdef struct SPAN_EIG{T} <: SPAN_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end
@kwdef struct SPAN_SPARSE{T} <: SPAN_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end
@kwdef struct SPAN_HOMO{T} <: SPAN_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end

solve(prob::SpanProblem) = solve(prob, default_alg(prob))
default_alg(prob::SpanProblem) = size(_get_b(prob), 2) == 1 ? SPAN_CHOL() : SPAN_EIG()

@testitem "Default solvers" begin
    using Random
    import AffineRayleighOptimization: SPAN_EIG, SPAN_CHOL
    Random.seed!(1234)
    prob_span = ConstrainedProblem(rand(2, 2), rand(1, 2), Span(rand(1)))
    # prob_vec = ConstrainedQuadraticFormProblem(rand(2, 2), rand(1, 2), Span(rand(1))
    @test solve(prob_span) ≈ solve(prob_span, SPAN_EIG()) || solve(prob_span) ≈ -solve(prob_span, SPAN_EIG())
    # @test solve(prob_vec) ≈ solve(prob_vec, SPAN_CHOL())
end


@testitem "Span incompatible" begin
    using Random
    import AffineRayleighOptimization: SPAN_GENEIG, SPAN_CHOL, SPAN_SPARSE, SPAN_HOMO
    Random.seed!(1234)
    prob = ConstrainedProblem(rand(1, 1), rand(1), Span(rand(1, 2)))
    for solver in [SPAN_GENEIG(), SPAN_CHOL(), SPAN_SPARSE(), SPAN_HOMO()]
        @test_throws ErrorException solve(prob, solver)
    end
end


function solve(prob::SpanProblem, alg::SPAN_GENEIG)
    size(_get_b(prob), 2) == 1 || error("$alg is incompatible with size(b, 2) > 1. Use SPAN_EIG() instead.")
    augC = hcat(prob.C, -_get_b(prob))
    N = nullspace(augC)
    Ntrunc = N[1:end-1, :]
    Nv = N[end, :]
    new_quadratic_form = Hermitian(Ntrunc' * prob.Q * Ntrunc)
    rhs_gen_eigen = Hermitian(Ntrunc' * Ntrunc)
    eigsol = eigen(new_quadratic_form, rhs_gen_eigen).vectors[:, 1] # smallest eigenvalue eigenvector maximizes
    unnormalized_sol = Ntrunc * eigsol
    return unnormalized_sol / dot(Nv, eigsol)
end

function solve(prob::SpanProblem, alg::SPAN_CHOL)
    size(_get_b(prob), 2) == 1 || error("$alg is incompatible with size(b, 2) > 1. Use SPAN_EIG() instead.")
    augC = hcat(prob.C, -_get_b(prob))
    N = nullspace(augC)
    Ntrunc = N[1:end-1, :]
    Nv = N[end, :]
    upper = cholesky(Ntrunc' * Ntrunc).U
    inv_upper = inv(upper)
    new_quadratic_form = Hermitian(inv_upper' * Ntrunc' * prob.Q * Ntrunc * inv_upper)
    eigsol = eigen(new_quadratic_form, 1:1).vectors[:, 1]
    unnormalized_sol = Ntrunc * inv_upper * eigsol
    return unnormalized_sol / dot(Nv, inv_upper * eigsol)
end

#https://www.cis.upenn.edu/~jshi/papers/supplement_nips2006.pdf
function solve(prob::SpanProblem, alg::Union{SPAN_EIG,SPAN_SPARSE}; normalize=size(_get_b(prob), 2) == 1)
    b = _get_b(prob)
    C_reduced = _get_C_reduced(prob, alg)
    homo_prob = HomogeneousProblem(prob.Q, C_reduced)
    sol = solve(homo_prob, SPAN_HOMO(alg.krylov_howmany, alg.krylov_kwargs))
    if normalize
        t = dot(b, prob.C, sol) / dot(b, b)
        return sol / t
    end
    return sol
end

function _get_C_reduced(prob::SpanProblem, alg::SPAN_EIG)
    b = _get_b(prob) # fetch b matrix/vector if prob.b is span/vector
    C = prob.C
    N = size(b, 1)
    Kb = I - b * b' / dot(b, b)
    _, inds_keep = _inds_to_remove_and_keep(b, N)
    return (Kb*C)[inds_keep, :] # instead of J * Kb * C where J = I(N)[inds_keep, :]
end

function _get_C_reduced(prob::SpanProblem, alg::SPAN_SPARSE)
    size(_get_b(prob), 2) == 1 || error("$alg is incompatible with size(b, 2) > 1. Use SPAN_EIG() instead.")
    b = _get_b(prob)
    C = prob.C
    inds_remove, inds_keep = _inds_to_remove_and_keep(b, length(b))
    ind_remove = only(inds_remove) # only one element, no support for b Span
    bk = b[ind_remove]
    Ck = transpose(C[ind_remove, :])
    return (C-(1/bk)*b*Ck)[inds_keep, :] # replace mult by J with slicing, see above
end

function _inds_to_remove_and_keep(b, N)
    inds_remove = partialsortperm(eachrow(b), 1:size(b, 2); by=norm, rev=true)
    inds_keep = map(!in(inds_remove), 1:N)
    return inds_remove, inds_keep
end

function solve(prob::SpanProblem, alg::SPAN_HOMO)
    iszero(_get_b(prob)) || error("b must be zero to use the algorithm SPAN_HOMO")
    solve(HomogeneousProblem(prob.Q, prob.C), SPAN_HOMO())
end
function solve(prob::HomogeneousProblem, alg::SPAN_HOMO)
    C = prob.C
    P, PQP = _get_P_PQP_homo(C, prob.Q)
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
    f = x -> dot(x, P, x) > 1 / 2
    return findfirst(f, eigvecs)
end

## Tests
@testitem "RayleighQuotientProblem" begin
    using LinearAlgebra, Random, SparseArrays
    import AffineRayleighOptimization: SPAN_GENEIG, SPAN_CHOL, SPAN_EIG, SPAN_SPARSE
    all_solvers = [SPAN_GENEIG(), SPAN_CHOL(), SPAN_EIG(), SPAN_SPARSE()]
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
        @test all(x -> x ≈ first(sols), sols[2:end])
    end
    Random.seed!(1234)
    #prob 1 (sol=b)
    Q = Diagonal(1:10)
    C = I(10)
    b = rand(10)
    prob = ConstrainedRayleighQuotientProblem(Q, C, b)
    test_prob_known_sol(prob, b)
    #prob 2 (span)
    b1 = [0.0, 2.0, zeros(8)...]
    b2 = [0.0, 0.0, 1.0, zeros(7)...]
    prob = ConstrainedRayleighQuotientProblem(Q, C, Span(b1, b2))
    test_prob_known_sol(prob, b1 / 2, [SPAN_EIG()])
    prob = ConstrainedRayleighQuotientProblem(Q, C, Span(b2, b1)) # change order
    test_prob_known_sol(prob, b1 / 2, [SPAN_EIG()])
    #prob 3
    C = ones(1, 10)
    b = [1.0]
    prob = ConstrainedRayleighQuotientProblem(Q, C, b)
    test_prob_known_sol(prob, [1.0, zeros(9)...])
    #prob 4 (random matrix)
    n = 20
    k = 10
    Q = Hermitian(rand(n, n))
    C = rand(k, n)
    b = rand(k)
    prob = ConstrainedRayleighQuotientProblem(Q, C, b)
    test_prob(prob)
    # prob 5 (positive definite matrix)
    M = Hermitian(rand(n, n))
    Q = M'M
    prob = ConstrainedRayleighQuotientProblem(Q, C, b)
    test_prob(prob)
    # prob 6 (sparse C)
    n = 1000
    k = 300
    Q = Hermitian(sprand(n, n, 1 / n))
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
    prob_sparse = ConstrainedRayleighQuotientProblem(Q, C, b)
    C_red_sparse = AffineRayleighOptimization._get_C_reduced(prob_sparse, SPAN_SPARSE())
    @test C_red_sparse isa SparseMatrixCSC
    sol = solve(prob_sparse, SPAN_SPARSE())
    prob_dense = ConstrainedRayleighQuotientProblem(Q, Matrix(C), Vector(b))
    test_prob_known_sol(prob_dense, sol)
    # prob 7 (sparse C with pos def Q)
    M = Hermitian(rand(n, n))
    Q = M'M
    prob_sparse = ConstrainedRayleighQuotientProblem(Q, C, b)
    # here we need to increase krylov_howmany to find the sol
    @test_throws ErrorException solve(prob_sparse, SPAN_SPARSE())
    sol = solve(prob_sparse, SPAN_SPARSE(krylov_howmany=10))
    Q_dense = Matrix(Q)
    prob_dense = ConstrainedRayleighQuotientProblem(Q_dense, Matrix(C), Vector(b))
    test_prob_known_sol(prob_dense, sol, [SPAN_CHOL(), SPAN_GENEIG()])
end

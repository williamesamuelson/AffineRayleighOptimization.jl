struct HomogeneousProblem{Q,C}
    Q::Q
    C::C
    function HomogeneousProblem(_Q, C)
        Q = Hermitian(_Q)
        return new{typeof(Q),typeof(C)}(Q, C)
    end
end

struct RayleighProblem{Q,C,B}
    Q::Q
    C::C
    b::B
    function RayleighProblem(_Q, C::CC, b::BB) where {CC,BB}
        Q = Hermitian(_Q)
        return new{typeof(Q),CC,BB}(Q, C, b)
    end
end

abstract type RAYLEIGH_ALG end
struct RAYLEIGH_CHOL <: RAYLEIGH_ALG end
struct RAYLEIGH_GENEIG <: RAYLEIGH_ALG end
const DEFAULT_SELECTVEC_EPS = 1e-15
@kwdef struct RAYLEIGH_EIG{T} <: RAYLEIGH_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end
@kwdef struct RAYLEIGH_SPARSE{T} <: RAYLEIGH_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end
@kwdef struct RAYLEIGH_HOMO{T} <: RAYLEIGH_ALG
    krylov_howmany::Int = 5
    krylov_kwargs::T = (;)
end

solve(prob::RayleighProblem) = solve(prob, default_alg(prob))
default_alg(prob::RayleighProblem) = size(prob.b, 2) == 1 ? RAYLEIGH_CHOL() : RAYLEIGH_EIG()

@testitem "Default solvers" begin
    using Random
    import AffineRayleighOptimization: RAYLEIGH_EIG, RAYLEIGH_CHOL
    Random.seed!(1234)
    prob = RayleighProblem(rand(2, 2), rand(1, 2), rand(1))
    @test solve(prob) ≈ solve(prob, RAYLEIGH_EIG()) || solve(prob) ≈ -solve(prob, RAYLEIGH_EIG())
end


@testitem "Span incompatible" begin
    using Random
    import AffineRayleighOptimization: RAYLEIGH_GENEIG, RAYLEIGH_CHOL, RAYLEIGH_SPARSE, RAYLEIGH_HOMO
    Random.seed!(1234)
    prob = RayleighProblem(rand(1, 1), rand(1), rand(1, 2))
    for solver in [RAYLEIGH_GENEIG(), RAYLEIGH_CHOL(), RAYLEIGH_SPARSE(), RAYLEIGH_HOMO()]
        @test_throws ErrorException solve(prob, solver)
    end
end


function solve(prob::RayleighProblem, alg::RAYLEIGH_GENEIG)
    ndims(prob.b) == 1 || error("$alg is incompatible with ndims(prob.b) != 1. Use RAYLEIGH_EIG() instead.")
    augC = hcat(prob.C, -prob.b)
    N = nullspace(augC)
    Ntrunc = N[1:end-1, :]
    Nv = N[end, :]
    new_quadratic_form = Hermitian(Ntrunc' * prob.Q * Ntrunc)
    rhs_gen_eigen = Hermitian(Ntrunc' * Ntrunc)
    eigsol = eigen(new_quadratic_form, rhs_gen_eigen).vectors[:, 1] # smallest eigenvalue eigenvector maximizes
    unnormalized_sol = Ntrunc * eigsol
    return unnormalized_sol / dot(Nv, eigsol)
end

function solve(prob::RayleighProblem, alg::RAYLEIGH_CHOL)
    ndims(prob.b) == 1 || error("$alg is incompatible with ndims(prob.b) != 1. Use RAYLEIGH_EIG() instead.")
    augC = hcat(prob.C, -prob.b)
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
function solve(prob::RayleighProblem, alg::Union{RAYLEIGH_EIG,RAYLEIGH_SPARSE}; normalize=size(prob.b, 2) == 1)
    b = prob.b
    C_reduced = _get_C_reduced(prob, alg)
    homo_prob = HomogeneousProblem(prob.Q, C_reduced)
    sol = solve(homo_prob, RAYLEIGH_HOMO(alg.krylov_howmany, alg.krylov_kwargs))
    if normalize
        t = dot(b, prob.C, sol) / dot(b, b)
        return sol / t
    end
    return sol
end

function _get_C_reduced(prob::RayleighProblem, alg::RAYLEIGH_EIG)
    b = prob.b # fetch b matrix/vector if prob.b is span/vector
    C = prob.C
    N = size(b, 1)
    Kb = I - b * b' / dot(b, b)
    _, inds_keep = _inds_to_remove_and_keep(b, N)
    return (Kb*C)[inds_keep, :] # instead of J * Kb * C where J = I(N)[inds_keep, :]
end

function _get_C_reduced(prob::RayleighProblem, alg::RAYLEIGH_SPARSE)
    ndims(prob.b) == 1 || error("$alg is incompatible with ndims(prob.b) != 1. Use RAYLEIGH_EIG() instead.")
    b = prob.b
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

function solve(prob::RayleighProblem, alg::RAYLEIGH_HOMO)
    iszero(prob.b) || error("b must be zero to use the algorithm RAYLEIGH_HOMO")
    solve(HomogeneousProblem(prob.Q, prob.C), RAYLEIGH_HOMO())
end
function solve(prob::HomogeneousProblem, alg::RAYLEIGH_HOMO)
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
@testitem "RayleighProblem" begin
    using LinearAlgebra, Random, SparseArrays
    import AffineRayleighOptimization: RAYLEIGH_GENEIG, RAYLEIGH_CHOL, RAYLEIGH_EIG, RAYLEIGH_SPARSE, RAYLEIGH_HOMO
    all_solvers = [RAYLEIGH_GENEIG(), RAYLEIGH_CHOL(), RAYLEIGH_EIG(), RAYLEIGH_SPARSE()]
    function test_prob_known_sol(prob, known_sol, solvers=all_solvers)
        for solver in solvers
            sol = solve(prob, solver)
            if ndims(prob.b) == 2
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
    prob = RayleighProblem(Q, C, b)
    test_prob_known_sol(prob, b)
    #prob 2 (span)
    b1 = [0.0, 2.0, zeros(8)...]
    b2 = [0.0, 0.0, 1.0, zeros(7)...]
    prob = RayleighProblem(Q, C, [b1 b2])
    test_prob_known_sol(prob, b1 / 2, [RAYLEIGH_EIG()])
    prob = RayleighProblem(Q, C, [b2 b1]) # change order
    test_prob_known_sol(prob, b1 / 2, [RAYLEIGH_EIG()])
    #prob 3
    C = ones(1, 10)
    b = [1.0]
    prob = RayleighProblem(Q, C, b)
    test_prob_known_sol(prob, [1.0, zeros(9)...])
    #prob 4 (random matrix)
    n = 20
    k = 10
    Q = Hermitian(rand(n, n))
    C = rand(k, n)
    b = rand(k)
    prob = RayleighProblem(Q, C, b)
    test_prob(prob)
    # prob 5 (positive definite matrix)
    M = Hermitian(rand(n, n))
    Q = M'M
    prob = RayleighProblem(Q, C, b)
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
    prob_sparse = RayleighProblem(Q, C, b)
    C_red_sparse = AffineRayleighOptimization._get_C_reduced(prob_sparse, RAYLEIGH_SPARSE())
    @test C_red_sparse isa SparseMatrixCSC
    sol = solve(prob_sparse, RAYLEIGH_SPARSE())
    prob_dense = RayleighProblem(Q, Matrix(C), Vector(b))
    test_prob_known_sol(prob_dense, sol)
    # prob 7 (sparse C with pos def Q)
    M = Hermitian(rand(n, n))
    Q = M'M
    prob_sparse = RayleighProblem(Q, C, b)
    # here we need to increase krylov_howmany to find the sol
    @test_throws ErrorException solve(prob_sparse, RAYLEIGH_SPARSE())
    sol = solve(prob_sparse, RAYLEIGH_SPARSE(krylov_howmany=10))
    Q_dense = Matrix(Q)
    prob_dense = RayleighProblem(Q_dense, Matrix(C), Vector(b))
    test_prob_known_sol(prob_dense, sol, [RAYLEIGH_CHOL(), RAYLEIGH_GENEIG()])
end

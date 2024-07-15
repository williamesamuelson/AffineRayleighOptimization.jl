using AffineRayleighOptimization, LinearAlgebra, SparseArrays
import AffineRayleighOptimization: RQ_EIG, RQ_SPARSE, RQ_GENEIG, RQ_CHOL
using BenchmarkTools, ProfileView, Plots

function generate_sparse_mats(sparsity, n, k)
    Q = Hermitian(rand(n,n))
    rc = RayleighQuotient(Q)
    b = sprand(k, 0.1)
    if iszero(b)
        b[1] = rand()
    end
    C = generate_sparse_fullrank_C(n, k, sparsity)
    return rc, C, b
end

function generate_sparse_fullrank_C(n, k, sparsity, maxit=1e4)
    C = zeros(k, n)
    for i in 1:maxit
        C = sprand(k, n, sparsity)
        fullrank = rank(C) == k
        fullrank && return C
    end
    error("maxit reached")
end

# remember to run first once to compile
function run_benchmark(sparsity, n, k)
    @time rc, C, b = generate_sparse_mats(sparsity, n, k)
    prob_sparse = ConstrainedRayleighQuotientProblem(rc, C, b)
    prob_dense = ConstrainedRayleighQuotientProblem(rc, Matrix(C), Vector(b))
    combs = collect(Base.product((:sparse, :dense), (RQ_SPARSE(), RQ_EIG()))) |> vec
    combs = vcat(combs, vec(collect(Base.product((:dense,), (RQ_CHOL(), RQ_GENEIG())))))
    times = zeros(length(combs))
    for (j, (prob, solver)) in enumerate(combs)
        if prob == :sparse
            times[j] =  @belapsed solve($prob_sparse, $solver)
        else
            times[j] =  @belapsed solve($prob_dense, $solver)
        end
    end
    return combs, times
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

function run_benchmarks(ns, ks, sparsity)
    return map((n, k)->run_benchmark(sparsity, n, k), ns, ks)
end

function benchmark_times(ns, ks, sparsity)
    datas = run_benchmarks(ns, ks, sparsity)
    times = mapreduce(data->data[2]', vcat, datas)
    #=labels = datas[1][1]=#
    return times
end

ns = [100, 200, 500, 1000]
ks = round.(Int, ns/100)
sparsities = [0.05, 0.01, 0.005]
#=[generate_sparse_mats(sparsity, ns[i], ks[i]) for sparsity in sparsities, i in eachindex(sparsities)]=#
times = []
for sp in sparsities
    time = benchmark_times(ns, ks, sp)
    display(time)
    push!(times, time)
end

pls = []
legend = ["RQ_SPARSE (s)" "RQ_SPARSE (d)" "RQ_EIG (s)" "RQ_EIG (d)" "RQ_CHOL" "RQ_GENEIG"]
for i in eachindex(sparsities)
    if i ≠ 1
        legend = false
    end
    p = plot(ns, times[i], labels=legend, marker=true, yscale=:log10, xscale=:log10, title="sparsity=$(sparsities[i])", legend=:topleft)
    push!(pls, p)
end
plot(pls..., layout=(2,2), dpi=300)
#=savefig("benchmark_k=n_over_100.png")=#

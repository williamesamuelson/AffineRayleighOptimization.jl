using AffineRayleighOptimization, LinearAlgebra, SparseArrays
import AffineRayleighOptimization: RAYLEIGH_EIG, RAYLEIGH_SPARSE, RAYLEIGH_GENEIG, RAYLEIGH_CHOL
using BenchmarkTools, ProfileView, Plots

function generate_sparse_mats(sparsity, n, k, maxit=1e4)
    Q = Hermitian(sprand(n, n, 1 / n))
    b = sparse(rand(k))
    C = generate_sparse_fullrank_C(n, k, sparsity, maxit)
    return Q, C, b
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

function get_combinations()
    combs = collect(Base.product((:sparse, :dense), (RAYLEIGH_SPARSE(), RAYLEIGH_EIG()))) |> vec
    return vcat(combs, vec(collect(Base.product((:dense,), (RAYLEIGH_CHOL(), RAYLEIGH_GENEIG())))))
end

# remember to run first once to compile
function run_benchmark(sparsity, n, k)
    @time Q, C, b = generate_sparse_mats(sparsity, n, k)
    prob_sparse = RayleighProblem(Q, C, b)
    prob_dense = RayleighProblem(Matrix(Q), Matrix(C), Vector(b))
    combs = get_combinations()
    times = zeros(length(combs))
    for (j, (prob, solver)) in enumerate(combs)
        if prob == :sparse
            times[j] = @belapsed solve($prob_sparse, $solver)
        else
            times[j] = @belapsed solve($prob_dense, $solver)
        end
    end
    return times
end

function run_benchmarks(ns, ks, sparsity)
    return map((n, k) -> run_benchmark(sparsity, n, k), ns, ks)
end

function benchmark_times(ns, ks, sparsity)
    datas = run_benchmarks(ns, ks, sparsity)
    times = mapreduce(data -> data', vcat, datas)
    return times
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

ns = [10, 100, 200, 500, 1000]
ks = round.(Int, ns / 10)
sparsities = [0.5, 0.1, 0.05, 0.01]
#=[generate_sparse_mats(sparsity, ns[i], ks[i], 1e9) for sparsity in sparsities, i in eachindex(ns)]=#
times = []
for sp in sparsities
    time = benchmark_times(ns, ks, sp)
    display(time)
    push!(times, time)
end

pls = []
display(get_combinations())
labels = ["RAYLEIGH_SPARSE (s)" "RAYLEIGH_SPARSE (d)" "RAYLEIGH_EIG (s)" "RAYLEIGH_EIG (d)" "RAYLEIGH_CHOL" "RAYLEIGH_GENEIG"]
for i in eachindex(sparsities)
    p = plot(ns, times[i], marker=true, yscale=:log10, xscale=:log10, title="sparsity=$(sparsities[i])", legend=false, xlabel="n", ylabel="time")
    push!(pls, p)
end
legend = plot(zeros(1,length(labels)), showaxis=false,lw=3, grid=false, label=labels)
plot(pls..., legend, layout=@layout([[A B; C D] E{0.25w}]))
#=savefig("benchmark_k=n_over_10.png")=#

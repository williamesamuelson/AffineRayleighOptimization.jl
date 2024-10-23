# AffineRayleighOptimization

[![Build Status](https://github.com/williamesamuelson/AffineRayleighOptimization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/williamesamuelson/AffineRayleighOptimization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/williamesamuelson/AffineRayleighOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/williamesamuelson/AffineRayleighOptimization.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

## Problems

This package contains solvers for the following optimization problems:

### Linearly Constrained Quadratic Form 
$$\min_x \quad x^\dagger Q x \quad \mathrm{s.t.} \quad Cx = b $$

Solve it by
```julia 
prob = QuadraticProblem(Q, C, b) 
solve(prob, alg)
```
where `alg` is `QF_BACKSLASH()` or `QF_LINEARSOLVE(solver)` where `solver` is a linear solver from `LinearAlgebra.jl` package.

### Affinely Constrained Quadratic Quotient 
$$\min_x \quad x^\dagger Q x \quad \mathrm{s.t.} \quad Cx = \mathrm{Span}(b_1, ... ,b_n) $$
If $ n=1 $ this problem is equivalent to the Rayleigh Quotient problem 
$$\min_x \quad \frac{x^\dagger Q x}{x^\dagger x} \quad \mathrm{s.t.} \quad Cx = b $$

Solve it by
```julia 
prob = QuadraticProblem(Q, C, Span(b)) 
#or alternatively 
prob = RayleighProblem(Q, C, b) 
solve(prob, alg)
```
where `alg ∈ [SPAN_GENEIG(), SPAN_CHOL(), SPAN_EIG(), SPAN_SPARSE(), SPAN_HOMO()]`
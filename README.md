# AffineRayleighOptimization
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://williamesamuelson.github.io/AffineRayleighOptimization.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://williamesamuelson.github.io/AffineRayleighOptimization.jl/dev/)
[![Build Status](https://github.com/williamesamuelson/AffineRayleighOptimization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/williamesamuelson/AffineRayleighOptimization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/williamesamuelson/AffineRayleighOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/williamesamuelson/AffineRayleighOptimization.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

## Problems

This package contains solvers for the following optimization problems:

### Linearly Constrained Quadratic Problem 
$$\min_x \quad x^\dagger Q x \quad \mathrm{s.t.} \quad Cx = b $$

Solve it by
```julia 
prob = QuadraticProblem(Q, C, b) 
solve(prob, [alg])
```
where `alg` is `QUADRATIC_BACKSLASH()` or `QUADRATIC_LINEARSOLVE(solver)` where `solver` is a linear solver from `LinearAlgebra.jl` package.

### Affinely Constrained Rayleigh Quotient Problem
$$\min_x \frac{\quad x^\dagger Q x}{x^\dagger x} \quad \mathrm{s.t.} \quad Cx = \mathrm{Span}(b_1, ... ,b_n) $$
If n=1 this problem is equivalent to problem 
$$\min_x \quad \frac{x^\dagger Q x}{x^\dagger x} \quad \mathrm{s.t.} \quad Cx = b $$

Solve it by
```julia 
prob = RayleighProblem(Q, C, b)
solve(prob, [alg])
```
where `alg ∈ [RAYLEIGH_GENEIG(), RAYLEIGH_CHOL(), RAYLEIGH_EIG(), RAYLEIGH_SPARSE(), RAYLEIGH_HOMO()]`. If `ndims(b)=1`, a solution such that `Cx=b` is returned. If `ndims(b)=2`, a solution such that `|x| = 1` and `Cx` is in the span of the columns of `b` is returned.

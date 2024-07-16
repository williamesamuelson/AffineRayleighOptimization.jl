# AffineRayleighOptimization

[![Build Status](https://github.com/williamesamuelson/AffineRayleighOptimization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/williamesamuelson/AffineRayleighOptimization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/williamesamuelson/AffineRayleighOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/williamesamuelson/AffineRayleighOptimization.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

## Problems

This package contains solvers for the following optimization problems:


1. Affinely Constrained Rayleigh Quotient 
$$\min_x \quad \frac{x^\dagger Q x}{x^\dagger x} \quad \mathrm{s.t.} \quad Cx = b $$

2. Affinely Constrained Quadratic Form 
$$\min_x \quad x^\dagger Q x \quad \mathrm{s.t.} \quad Cx = b $$

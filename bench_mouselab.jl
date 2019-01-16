cd("StrategyDiscovery/Journal/julia")
include("mouselab.jl")
using Profile
using Test
const prm = Params()
const p = Problem(prm)
const b = Belief(p)
const gamble_dists = gamble_values(b)
const μ = mean.(gamble_dists)[:]
const pol = Policy([0, 0, 0, 1.])

#%%
println("=============== bench_voi1 ===============")
function bench_voi1()
    for i in 1:100000
        voi1(b, 1, μ)
    end
end
bench_voi1()
@time bench_voi1()
# @code_warntype voi1(b, 1)
# @inferred voi1(b, 1)
# @profiler bench_voi1_opt()

#%%
println("=============== bench_voi_gamble ===============")
function bench_voi_gamble()

    for i in 1:100000
        voi_gamble(b, 1, gamble_dists, μ)
    end
end
bench_voi_gamble()
@time bench_voi_gamble()
# @code_warntype voi_gamble(b, 1, gamble_dists, μ)
# @inferred voi_gamble(b, 1, gamble_dists, μ)
# @profiler bench_voi_gamble()

#%%
# include("mouselab.jl")
println("=============== bench_vpi ===============")
# const SAMPLES = randn(N_SAMPLE)

# const S2 = randn(N_SAMPLE, length(    gamble_dists))
# const S4 = randn(length(gamble_dists), N_SAMPLE)


X = μ .+ σ .* S3
maximum(X, dims=1)

[i + j for i=1:5, j=1:5]
function vpi(b::Belief, gamble_dists::Vector{Normal{Float64}}, μ::Vector{Float64})::Float64
    # σ = std.(gamble_dists)
    # μ .+ σ .* S4
    # mean(maximum(X, dims=1))
    # samples = [d.μ + d.σ * s for d in gamble_dists, s in SAMPLES]
    # mean(maximum(samples, dims=1))
    # samples = [d.μ + d.σ * s for s in SAMPLES, d in gamble_dists]
    # mean(maximum(samples, dims=2))


    # mean(maximum(ss) for ss in samples) - maximum(μ)
    # mean(max.(samples...)) - maximum(μ)
end

function bench_vpi(;n=1000)
    for i in 1:n
        vpi(b, gamble_dists, μ)
    end
end

bench_vpi()
@time bench_vpi()
# @code_warntype vpi(b)
# @inferred vpi(b)


#%%
println("=============== bench_pol ===============")
function bench_pol(b)
    for i in 1:10
        pol(b)
    end
end
bench_pol(b)
@time bench_pol(b)
# @profiler bench_pol(b)

#%%
include("mouselab.jl")
println("=============== bench_roll ===============")
function bench_roll(;n=10)
    r = 0.
    for i in 1:n
        r += rollout(pol, b).reward
    end
    r / n
end
bench_roll()
@time bench_roll();

using BenchmarkTools
bench = @benchmark rollout(pol, b)
display(bench)
# @profiler bench_roll(b)


#%%
function observe_all(b)
    b = deepcopy(b)
    for c in unobserved(b)
        observe!(b, c)
    end
    b
end

#
#
# @time [features(b; no_vpi=false) for i in 1:1000];
# @time [features(b; no_vpi=true) for i in 1:1000];
# @time features(observe_all(b));
# @time features(b);
#
# const N_CELL= 28
# function rand_belief()
#     clicks = sample(1:N_CELL, rand(1:N_CELL), replace=false)
#     b1 = deepcopy(b)
#     for c in clicks
#         observe!(b1, c)
#     end
#     b1
# end
#
# b1 = rand_belief()
# [voi1(b1, c) for c in 1:N_CELL]
#
#
# #%%
#
#
#
# gamble_dists = gamble_values(b)
# μ = mean.(gamble_dists)[:]
#
# function test()
#     @time vpi(b, gamble_dists, μ)
#     @time samples = [rand(d, N_SAMPLE) for d in gamble_dists]
#     @time mean(max.(samples...)) - maximum(μ)
# end
# @time vpi(b)
#
#
# #%%
# gamble_dists = gamble_values(b)
# μ = mean.(gamble_dists)[:]
# const RANDN = randn(7, N_SAMPLE)
# function vpi2(b, gamble_dists, μ)
#     μ_g = [d.μ for d in gamble_dists]
#     σ_g = [d.σ for d in gamble_dists]
#     mean(maximum(μ_g .+ RANDN .* σ_g, dims=1)) - maximum(μ)
# end
# @time [vpi2(b, gamble_dists, μ) for i in 1:100];
# @time [vpi(b, gamble_dists, μ) for i in 1:100];
#
#
# μ_g = [d.μ for d in gamble_dists]
# σ_g = [d.σ for d in gamble_dists]
#
#
# x = randn(100)
# y = randn(100)
# d = gamble_dists[1]
# x = randn(10)
# y = randn(10)
# max.(x, y)
# max(x, y)
#
# vpi(b)

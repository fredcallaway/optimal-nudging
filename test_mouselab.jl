using Test
include("mouselab.jl")

const N_PROBLEM = 100
const N_GAMBLE = 7
const N_OUTCOME = 4
const N_CELL = N_GAMBLE * N_OUTCOME
const CELLS = reshape(1:N_CELL, N_OUTCOME, N_GAMBLE)
const N_SAMPLE = 10000

#%% ========== Helpers ==========
rand_problem() = Problem(Params(
    reward_dist=Normal(randn(), rand()+0.01),
    compensatory=rand([true, false])
))

function rand_belief()
    p = rand_problem()
    b = Belief(p)
    clicks = sample(1:N_CELL, rand(1:N_CELL), replace=false)
    for c in clicks
        observe!(b, c)
    end
    b
end

function observe_all(b)
    b = deepcopy(b)
    for c in unobserved(b)
        observe!(b, c)
    end
    b
end
function observe_all(b, p)
    b = deepcopy(b)
    for c in unobserved(b)
        observe!(b, p, c)
    end
    b
end

"Estimates the value of f() ± 3*SEM"
function mc_est(f::Function; n_sample=N_SAMPLE)
    samples = [f() for i in 1:n_sample]
    sem = std(samples) / n_sample
    return mean(samples), max(1e-5, 3sem)
end

@testset "unobserved" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        for c in unobserved(b)
            @test !observed(b, c)
        end
        @test rand_belief() |> observe_all |> unobserved |> isempty
    end
end

function observe_gamble(b, gamble)
    b = deepcopy(b)
    for c in CELLS[:, gamble]
        if !observed(b, c)
            observe!(b, c)
        end
    end
    b
end

function observe_outcome(b, outcome)
    b = deepcopy(b)
    for c in CELLS[outcome, :]
        if !observed(b, c)
            observe!(b, c)
        end
    end
    b
end

#%% ========== Tests ==========

@testset "term_reward" begin
    for i in 1:N_PROBLEM
        p = rand_problem()
        b = Belief(p)
        @test term_reward(b) ≈ p.prm.reward_dist.μ
    end
    for i in 1:N_PROBLEM
        p = rand_problem()
        b = observe_all(Belief(p), p)
        @test term_reward(b) ≈ maximum(p.weights' * p.matrix)
    end
end

@testset "voi1" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        while isempty(unobserved(b))
            b = rand_belief()
        end
        c = rand(unobserved(b))
        @test !observed(b, c)
        base = term_reward(b)
        mcq = mean(term_reward(observe(b, c)) for i in 1:N_SAMPLE)
        v = voi1(b, c)
        @test v >= 0
        @test mcq - base ≈ v atol=0.01
    end
    # b = Belief(rand_problem())
    # observe!(b, 1)
    # voi1(b, 1) == 0
end


#%%

@testset "vpi" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        base = term_reward(b)
        est, ε = mc_est(()-> term_reward(observe_all(b)) - base)
        @test est ≈ vpi(b) atol=ε
    end
end


@testset "voi_gamble" begin
    for i in 1:N_PROBLEM
        # b = rand_belief()
        b = rand_belief()
        base = term_reward(b)
        g = rand(1:N_GAMBLE)
        est, ε = mc_est(()-> term_reward(observe_gamble(b, g)) - base)
        @test est ≈ voi_gamble(b, g) atol=ε
    end
end

@testset "voi_outcome" begin
    for i in 1:N_PROBLEM
        # b = rand_belief()
        b = rand_belief()
        base = term_reward(b)
        o = rand(1:N_OUTCOME)
        est, ε = mc_est(()-> term_reward(observe_outcome(b, o)) - base)
        @test est ≈ voi_outcome(b, o) atol=ε
    end
end

@testset "features" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        φ = features(b)
        int, v1, vg, vo, vp = [reshape(φ[i, :], N_OUTCOME, N_GAMBLE) for i in 1:5]
        unobs = .!(getfield.(b.matrix, :σ) .== 1e-20)
        @test all(int[unobs] .== -1)
        @test length(unique(vp[unobs])) <= 1
        @test map(1:N_GAMBLE) do i
            all(length(unique([-1e10; vg[:, i]])) <= 2)
        end |> all
    end
end

@testset "policy" begin
    pol = Policy([0; ones(4) ./ 4])
    rollout(pol, rand_problem())

#%% ========== Scratch ==========

# @testset "meta greedy" begin
#     meta_greedy = Policy([0., 1., 0, 0, 0])
#     for i in 1:N_PROBLEM
#         b = rand_belief()
#         est, ε = mc_est(()-> rollout(meta_greedy, b).reward; n_sample=1000)
#         @test est >= (term_reward(b) - ε)
#     end
# end

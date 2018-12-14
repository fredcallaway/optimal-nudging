using Distributed
using JLD
@everywhere include("mouselab.jl")
import Random
using Dates: now
using Printf: @printf
using PyCall
using LatinHypercubeSampling: LHCoptim

@pyimport skopt
# Optimizer methods
ask(opt)::Vector{Float64} = opt[:ask]()
tell(opt, x::Vector{Float64}, fx::Float64) = opt[:tell](Tuple(x), fx)

function x2θ(x, voi_features)
    cost_weight = x[1]
    ws = diff([0; sort(collect(x[2:end])); 1])
    voi_weights = zeros(4)
    for (f, w) in zip(voi_features, ws)
        voi_weights[f] = w
    end
    [cost_weight; voi_weights]
end

function max_cost(prm::Params)
    p = Problem(prm)
    b = Belief(p)
    θ = Float64[1, 0, 0, 0, 1]
    computes() = Policy(θ)(b) != TERM

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 100
    while computes()
        θ[1] += step_size
    end
    θ[1]
end

function avg_reward(prm, θ; n_roll=100)
    reward = @distributed (+) for i in 1:n_roll
        π = Policy(θ)
        rollout(π, Problem(prm), max_steps=200).reward
    end
    reward / n_roll
end

function optimize(prm::Params; voi_features=1:4, seed=0, n_iter=100, n_roll=1000, verbose=false)
    function loss(x; nr=n_roll)
        reward, secs = @timed avg_reward(prm, x2θ(x, voi_features); n_roll=n_roll)
        verbose && @printf "reward = %.3f   seconds = %.3f\n" reward secs
        flush(stdout)
        - reward
    end
    bounds = [(0., max_cost(prm)); repeat([(0., 1.)], length(voi_features)-1)]
    opt = skopt.Optimizer(bounds, random_state=seed)

    # Choose first 25% of points by Latin Hypersquare sampling.
    upper_bounds = [b[2] for b in bounds]
    n_latin = max(2, cld(n_iter, 4))
    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1]
    for i in 1:n_latin
        x = latin_points[i, :] ./ n_latin .* upper_bounds
        tell(opt, x, loss(x))
    end

    # Bayesian optimization.
    for i in 1:(n_iter - n_latin)
        x = ask(opt)
        tell(opt, x, loss(x))
    end

    # Cross validation.
    top_x = opt[:Xi][sortperm(opt[:yi])][1:cld(n_iter, 5)]  # top 20%
    top_θ = [x2θ(x, voi_features) for x in top_x]
    top_loss = loss.(top_x; nr=n_roll*10)
    perm = sortperm(top_loss)
    top_θ = top_θ[perm]
    top_loss = top_loss[perm]
    return (theta=top_θ[1], reward=-top_loss[1],
            top_theta=top_θ, top_reward=-top_loss,
            voi_features=voi_features)
end

function name(prm::Params)
    join(map(string, (
        prm.n_gamble,
        prm.n_outcome,
        prm.reward_dist.μ,
        prm.reward_dist.σ,
        prm.compensatory,
        prm.cost
    )), "-")
end


import JSON
read_args(file) = Dict(Symbol(k)=>v for (k, v) in JSON.parsefile(file))



function main(prm::Params; jobname="none", seed=nothing, opt_args...)
    println(prm)
    if seed == nothing
        seed = Int(rand(1:1e8))
    end
    println("Seed: ", seed)
    println("Running with $(length(workers())) workers.")
    target = "runs/$(jobname)/results"
    mkpath(target)
    Random.seed!(seed)
    @time result = optimize(prm; seed=seed, opt_args...)
    println("THETA: ", result.theta)
    println("REWARD: ", result.reward)
    file = "$(target)/opt-$seed-$(name(prm)).jld"
    result = Dict(pairs(result))
    result[:prm] = prm
    result[:time] = now()
    save(file, "opt_result", result)
    println("Wrote $file")
    result
end

function main(file::String; opt_args...)
    args = read_args(file)
    seed = pop!(args, :seed, nothing)
    jobname = pop!(args, :job_name, "none")
    stakes = pop!(args, :stakes)
    voi_features = pop!(args, :voi_features)
    args[:reward_dist] = exp_dist(stakes)
    prm = Params(;args...)
    main(prm; jobname=jobname, seed=seed, verbose=true, voi_features=voi_features, opt_args...)
end

if !isempty(ARGS)
    if ARGS[1] == "test"
        main("runs/test/jobs/1.json"; n_iter=4, n_roll=4)
    else
        job_group, job_id = ARGS
        main("runs/$job_group/jobs/$job_id.json")
    end
end

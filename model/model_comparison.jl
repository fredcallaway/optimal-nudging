include("nudging_base.jl")
include("suggest_new.jl")
include("likelihood.jl")
include("data.jl")

trials = load_trials("pilot5_supersize");

naive_likes = @showprogress map(trials) do t
    likelihood(t; naive=true)
end

savvy_likes = @showprogress map(trials) do t
    likelihood(t; naive=false)
end

logK = marginal_individual(savvy_likes, choices) .- marginal_individual(naive_likes, choices)
log10K = logK ./ log(10)

using Printf
begin
    @printf "%.1f%% savvy\n" 100mean(log10K .> 1/2)
    @printf "%.1f%% naive\n" 100mean(log10K .< -1/2)
    @printf "log₁₀K = %.1f\n" sum(log10K)
end

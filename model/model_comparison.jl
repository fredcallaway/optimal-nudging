@everywhere include("nudging_base.jl")
@everywhere include("suggest_new.jl")
@everywhere include("data.jl")
@everywhere include("likelihood.jl")

using ProgressMeter

function num_optimal(t::Trial)
    s = State(t)
    b = Belief(s)
    pol = MetaGreedy(s.m)
    v = voc(pol, b)
    length(findall(isequal(maximum(v)), v))
end

trials = load_trials("supersize_data");
# filter!(t-> num_optimal(t) <= 12, trials)
grouped_trials = collect(group(t->t.participant_id, trials));

marginals = @showprogress pmap(grouped_trials) do trials
    naive = map(trials) do t
        likelihood(t; naive=true)
    end
    savvy = map(trials) do t
        likelihood(t; naive=false)
    end
    both = (;naive, savvy)
    choices = getfield.(trials, :choice)
    map(both) do like
        marginal_logp(like, choices)
    end
end

# # %% --------
# marginal = map(grouped_trials, all_like) do (trials, like)
#     choices = getfield.(trials, :choice)
#     map(like) do ll
#         marginal_logp(ll, choices)
#     end
# end
# %% --------
M = invert(marginals)
logK = M.savvy - M.naive
log10K = (M.savvy .- M.naive) ./ log(10)

using Printf
begin
    @printf "%.1f%% savvy\n" 100mean(log10K .> 1/2)
    @printf "%.1f%% naive\n" 100mean(log10K .< -1/2)
    @printf "log₁₀K = %.1f\n" sum(log10K)
end

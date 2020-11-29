using SplitApplyCombine

"Expected maximum of a distribution and and a consant."
function emax(d::Distribution, c::Float64)
    p_improve = 1 - cdf(d, c)
    p_improve < 1e-10 && return c
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)

function dc_plans(b::Belief; cost=nothing)
    cost = cost == nothing ? b.m.cost : cost
    n_feature, n_option = size(b.matrix)
    current_best = maximum(choice_values(b))
    plans = map(1:n_option) do option
        ranked_clicks = sortperm([-d.σ for d in b.matrix[:, option]])
        cv = competing_value(µ, option)
        map(1:n_feature) do n_click
            chosen = ranked_clicks[1:n_click]
            new_dist = Normal(0, 1e-20)
            for i in 1:n_feature
                d = b.matrix[i, option]
                new_dist += (i in chosen ? d : d.μ)
            end
            voi = emax(new_dist, cv) - current_best
            voc = voi - cost * n_click
            clicks = chosen .+ (n_feature * (option - 1))  # convert to full matrix indices
            (voc=voc, clicks=clicks)
        end
    end |> flatten
    push!(plans, (voc=0., clicks=[⊥]))
    plans
end

function select_option(b::Belief)
    plans = dc_plans(b)
    voc, clicks = invert(plans)
    voc .+= 1e-10 .* rand(length(voc))
    v, i = findmax(voc)
    clicks[i]
    # v <= 0 ? [⊥] : clicks[i]
end


struct DCPolicy <: Policy
    m::MetaMDP
    replan::Bool
    stack::Vector{Int}
end
DCPolicy(m::MetaMDP, replan::Bool=true) = DCPolicy(m, replan, [])
(π::DCPolicy)(b::Belief) = begin
    if π.replan
        select_option(b)[1]
    else
        if isempty(π.stack)
            append!(π.stack, reverse(select_option(b)))
        end
        pop!(π.stack)
    end
end

# struct EpsilonGreedy <: Policy
#     π::Policy
#     ε::Float64
# end
# (π::EpsilonGreedy)(b::Belief) = begin
#     if isempty(π.π.stack) && rand() < ε
#         return rand([unobserved(b); 0])
#     else
#         return π.π(b)
#     end
# end

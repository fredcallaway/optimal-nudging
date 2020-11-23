using SplitApplyCombine

"Expected maximum of a distribution and and a consant."
function emax(d::Distribution, c::Float64)
    p_improve = 1 - cdf(d, c)
    p_improve < 1e-10 && return c
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)

function dc_options(b::Belief; cost=nothing)
    cost = cost == nothing ? b.m.cost : cost
    n_outcome, n_gamble = size(b.matrix)
    μ = mean.(gamble_values(b))
    options = map(1:n_gamble) do gamble
        ranked_clicks = sortperm([-d.σ for d in b.matrix[:, gamble]])
        cv = competing_value(µ, gamble)
        map(1:n_outcome) do n_click
            chosen = ranked_clicks[1:n_click]
            new_dist = Normal(0, 1e-20)
            for i in 1:n_outcome
                d = b.matrix[i, gamble]
                new_dist += (i in chosen ? d : d.μ)
            end
            voi = emax(new_dist, cv) - maximum(μ)
            voc = voi - cost * n_click
            clicks = chosen .+ (n_outcome * (gamble - 1))
            (voc=voc, clicks=clicks)
        end
    end |> flatten
    push!(options, (voc=0., clicks=[⊥]))
    options
end

function select_option(b::Belief)
    options = dc_options(b)
    voc, clicks = invert(options)
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

using Printf
function describe_vec(x::Vector)
    @printf("%.3f ± %.3f  [%.3f, %.3f]\n", juxt(mean, std, minimum, maximum)(x)...)
end

Base.map(f, d::AbstractDict) = [f(k, v) for (k, v) in d]
valmap(f, d::AbstractDict) = Dict(k => f(v) for (k, v) in d)
valmap(f) = d->valmap(f, d)
keymap(f, d::AbstractDict) = Dict(f(k) => v for (k, v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]

type2nt(p) = (;(v=>getfield(p, v) for v in fieldnames(typeof(p)))...)
fields(p) = [getfield(p, v) for v in fieldnames(typeof(p))]

function Base.get(collection, key)
    v = get(collection, key, "__missing__")
    if v == "__missing__"
        error("Key $key not found in collection!")
    end
    v
end

Base.get(key) = (collection) -> get(collection, key)

function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

macro ifundef(exp)
    local e = :($exp)
    isdefined(Main, e.args[1]) ? :($(e.args[1])) : :($(esc(exp)))
end

function mutate(x::T; kws...) where T
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end



# import Serialization: serialize
# function serialize(s::String, x)
#     open(s, "w") do f
#         Serialization.serialize(f, x)
#     end
# end
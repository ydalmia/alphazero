# using Base: Unicode
# RUN SETTINGS: julia --threads 8
# may also want: --optimize=3 --math-mode=fast --inline=yes --check-bounds=no
using Chess: isdraw, ischeckmate, Board, domove!, tostring, pprint
using Flux
using StatsBase

include("translation.jl") # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for storing monte carlo searc results
include("neuralnet.jl") # the brain that helps monte carlo focus its search

const encoder, _ = alphazero_encoder_decoder()

function mcts(s::Board, node::TreeNode)
    if isdraw(s)
        backpropagate!(Float32(0.0), node)
    elseif ischeckmate(s)
        backpropagate!(Float32(1.0), node)
    elseif node.isleaf
        node.isleaf = false
        r = expand!(node, s)
        backpropagate!(r, node)
    else
        W, N, P = children_stats(node)            
        child = node.children[argmaxUCT(W, N, P)]
        domove!(s, child.A)
        mcts(s, child)
    end
end

function expand!(node::TreeNode, s::Board)
    base = f(alphazero_rep(s)) # ask neural net for policy and value
    p = Array(policy(base))
    v = Array(value(base))
    a, vp = validpolicy(s, p, encoder)

    a = map(a) do x
        if !ispromotion(x) && ptype(pieceon(s, from(x))) == PAWN && rank(from(x)) == SS_RANK_7 
            # actually a queen promotion
            return Move(from(x), to(x), QUEEN)
        else
            return x
        end
    end

    # if the action recommended is to promote a pawn, then convert it to a QUEEN# unless it recommends something else
    node.children = [TreeNode(a, vp, node) for (a, vp) in zip(a, vp)]
    return v[1, 1] # heuristic r
end

function backpropagate!(r::Float32, node::TreeNode)
    while node !== nothing
        node.W += r 
        node.N += Float32(1.0)
        node = node.parent
        r = -r
    end
end

function argmaxUCT(W::Vector{Float32}, N::Vector{Float32}, P::Vector{Float32}, c=2.0)
    Q = [N==0.0 ? 0.0 : W/N for (W, N) in zip(W, N)]
    U = c * P ./ (1 .+ N) * √sum(N)
    UCT = Q .+ U
    return argmax(UCT)
end

function validpolicy(s::Board, p::Array, encoder::Dict)
    a = moves(s) # list of valid moves, according to chess rules

    p = reshape(p, (8, 8, 88)) # neural net spits p out as a 1-d vector
    nmoves = length(a)

    vp = Vector{Float32}(undef, nmoves)
    for i in 1:nmoves
        row, col, plane = alphazero_rep(a[i], encoder) # translate uci to az encoding
        vp[i] = p[row, col, plane] # extract the policy values from nnet output
    end

    vp = vp ./ sum(vp) # normalize the valid policy
    return a, vp
end

function simulate(root, s, nsims)
    for _ in 1:nsims
        mcts(deepcopy(s), root)
    end
end

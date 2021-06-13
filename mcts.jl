# RUN SETTINGS: julia --threads 8
# may also want: --optimize=3 --math-mode=fast --inline=yes --check-bounds=no
using Chess: isdraw, ischeckmate, Board, domove!, tostring, pprint

include("translation.jl") # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for storing monte carlo searc results
include("neuralnet.jl") # the brain that helps monte carlo focus its search

function simulate(root::TreeNode, s::Board, nsims::Int64)
    for _ in 1:nsims
        mcts(deepcopy(s), root) # deep copy bc making moves modifies board (s)
    end
end

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
    x = alphazero_rep(s) # convert board to features
    x = reshape(x, (size(x)..., 1)) # batch size 1, batch dim comes last in flux

    base = f(x) # ask neural net for policy and value
    p = policy(base)[:, 1] # p is a dense output, and has batch size 1
    v = value(base)[1, 1] # v is a single scalar value, and has batch size 1
    a, vp = valid_policy(s, p)

    node.children = [TreeNode(a, vp, node) for (a, vp) in zip(a, vp)]
    return v
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
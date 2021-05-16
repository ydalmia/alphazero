using Pkg
Pkg.add("Chess") # Chess: https://romstad.github.io/Chess.jl/dev/manual/
using Chess

function mcts(s, node)
    if isdraw(s)
        backpropagate(0, node) # no reward, but must still update counts

    elseif ischeckmate(s)
        backpropagate(-1, node) # prev player checkmated us

    elseif node == missing
        p, v = fθ(s)
        nmoves, a, p = validpolicy(s, p)

        # a returns a chess
        node.cA, node.cP = a, p
        node.cN = node.cQ = zeros(nmoves)
        node.cNode = Vector{TreeNode}(missing, nmoves)

        backpropagate(v, node)

    else
        a = node.A[argmaxUCT(node.W, node.N, node.P)]
        domove!(s, a)
        mcts(s, node.children[a])
    end
end


mutable struct TreeNode
    isleaf::Bool
    parent::TreeNode
    cW::Vector{Float16}
    cN::Vector{Float16}
    cA::Vector{Move}
    cP::Vector{Float16}
    cNode::Vector{TreeNode}
end

node = TreeNode(true, nothing, nothing, nothing, nothing, nothing, nothing)
b = startboard()
mcts(b, node)

function backpropagate(r, node)
    while node != NULL
        # update count and total reward
        node.N += 1
        node.W += r

        # tree switches between white & black
        node = node.parent
        r = -r
    end
end


function argmaxUCT(W, N, P, c=2.0)
    Q = W / N
    U = c * P / (1 + N) * √sum(N)
    return argmax(Q + U)
end


function validpolicy(s, p)
    a = genmoves(s)
    nmoves = length(a)

    # valid policy
    vp = [p[row, col, plane] for (row, col, plane) in alphazero_rep.(a)]
    vp /= sum(vp)

    return nmoves, a, vp
end

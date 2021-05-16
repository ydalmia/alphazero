using Pkg
Pkg.add("Chess") # Chess: https://romstad.github.io/Chess.jl/dev/manual/
using Chess

function mcts(s, node)
    if isdraw(s)
        backpropagate(0, node) # no reward, but must still update counts

    elseif ischeckmate(s)
        backpropagate(-1, node) # prev player checkmated us

    elseif node.isleaf
        node.isleaf = false
        p, v = fθ(s)
        nmoves, a = validpolicy(s)

        node.cA, node.cP = a, p
        node.cN = node.cW = zeros(nmoves)
        node.cNode = Vector{TreeNode}(TreeNode, nmoves)

        backpropagate(v, node)

    else
        a = node.cA[argmaxUCT(node.cW, node.cN, node.cP)]
        sp = !domove(s, a)
        mcts(sp, node.children[a])
    end
end


mutable struct TreeNode
    isleaf
    parent
    cW
    cN
    cA
    cP
    cNode
end

node = TreeNode(true, nothing, nothing, nothing, nothing, nothing, nothing)

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

function fθ(s)
    return rand(Float16, (8,8,88)), rand(Float16, (1))
end


function validpolicy(s, p)
    a = genmoves(s)
    nmoves = length(a)

    # valid policy
    vp = [p[row, col, plane] for (row, col, plane) in alphazero_rep.(a)]
    vp /= sum(vp)

    return nmoves, a, vp
end


board = fromfen("R2R1rk1/5p1p/4nQpP/4p2q/3pP3/r1pP3P/2B2PP1/6K1 w - - 0 1")
mcts(board, node)
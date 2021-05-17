using Pkg

# Chess: https://romstad.github.io/Chess.jl/dev/manual/
# This is a general chess programming package, used throughout the project
Pkg.add("Chess")
using Chess

# utility functions for translating between alphazero neural network
# and UCI (universal chess interface) notation used in Chess.jl package
include("representation.jl")

function fθ(s)
    p = rand(Float16, (8, 8, 88))
    v = rand(Float16, 1)
    return p, v
end

function mcts(s, node, idx, encoder)
    node.idx = idx
    if isdraw(s)
        # we drew, so no reward, but still need to update counts to show
        # that it would be better to pursue other routes
        backpropagate(0, node)

    elseif ischeckmate(s)
        # prev player checkmated us, negative reward
        backpropagate(1, node)

    elseif node.isleaf
        # we have now explored it, no longer a leaf
        node.isleaf = false
        # ask neural network for recommended policy
        # and the value of current state
        p, v = fθ(s)
        # extract the moves which are legal, and renormalize
        nmoves, a, p = validpolicy(s, p, encoder)
        p /= sum(p)
        # store the actions we can take in a
        # store the policy recommendation for each of those actions
        node.cA, node.cP = a, p
        # we have no statistics on how good any of those actions are
        node.cN = node.cW = zeros(Float16, nmoves)
        # we haven't explored the children yet!
        node.children = [TreeNode() for _ in 1:nmoves]
        # update parents with the reward
        backpropagate(v, node)

    else
        # greedily choose action that maximizes the UCT score
        # (the UCT balances exploration and exploitation)
        idx = argmaxUCT(node.cW, node.cN, node.cP)
        a = node.cA[idx]
        # use the chosen action to transition to new state
        # (modify state in place to avoid allocation)
        # (domove! returns a move that undoes the move, but we don't need it)
        domove!(s, a)
        # expand the tree by searching further along this path
        # since this was a promising path according to UCT score
        mcts(s, node.children[a], idx, encoder)
    end
end


function backpropagate(r, node)
    while node != undef && node.parent != undef
        # update count and total reward
        idx = node.idx
        node.parent.cN[idx] += 1
        node.parent.cW[idx] += r

        # we flip reward value sign since at each level, the tree switches
        # between white & black, and a white win means a black loss
        # and vice versa
        node = node.parent
        r = -r
    end
end


function argmaxUCT(W, N, P, c=2.0)
    # alphazero's modified version of UCT, see the paper for more info
    # but broadly, it balances between exploration and exploitation,
    # but as time goes on, we favor exploitation instead of exploration
    Q = W / N
    U = c * P / (1 + N) * √sum(N)
    return argmax(Q + U)
end


function validpolicy(s, p, encoder)
    # list of valid moves, according to chess rules
    a = moves(s)
    nmoves = length(a)

    println(p)
    # extract the policy values for only the valid actions
    # by converting uci moves to alphazero plane moves
    # and then reading the value from the planes
    vp = Vector{Float16}(undef, nmoves)
    for i in 1:nmoves
        row, col, plane = alphazero_rep(a[i], encoder)
        println("row: ", row, "col: ", col, "plane: ", plane)
        vp[i] = p[row, col, plane]
    end

    return nmoves, a, vp
end



# MCTS Tree
# broadly, each node in the tree stores the statistics of the children,
# to improve spatial locality (and therefore, increase cache hits).
# For example, when calculating UCT scores,
# you need to calculate the score for each child, which means accessing
# a pointer for 30+ possible child states, which would be very slow.
# By storing the children's statistics in an array, not only is there no
# reference chasing, we can also vectorize the UCT calculation
mutable struct TreeNode
    # at the bottom of the mcts search tree?
    isleaf::Bool
    # store the children's statistics (i.e. their Q, N, P values)
    cW::Vector{Float16}
    cN::Vector{Float16}
    cP::Vector{Float16}
    # UCI move that, when applied to a board,
    # transition to the child board state
    cA::Vector{Move}
    # store reference to children
    children::Vector{TreeNode}
    # store reference to parent
    parent::TreeNode
    idx::UInt8
    # default constructor for self referential objects, see below link:
    # https://discourse.julialang.org/t/how-to-create-tree-from-struct/39372/3
    TreeNode() = new(true)
end


function playgame()
    encoder, decoder = alphazero_encoder_decoder()
    s = startboard()
    root = TreeNode()
    mcts(s, root, 0, encoder)
end

println(playgame())

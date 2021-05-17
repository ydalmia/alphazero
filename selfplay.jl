using Pkg

Pkg.add("Chess")    # Chess: https://romstad.github.io/Chess.jl/dev/manual/
using Chess         # General chess package, used throughout the project


include("translation.jl")   # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for monte carlo searches
include("neuralnet.jl") # the brain that helps monte carlo focus its search


function mcts(s, node, encoder)
    if isdraw(s) # we drew, so no reward, but still need to update counts to show
        backpropagate(0, node)

    elseif ischeckmate(s) # prev player checkmated us, negative reward
        backpropagate(-1, node)

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
        for idx in 1:nmoves
            node.children[idx].idx = idx
            node.children[idx].parent = node
        end

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
        mcts(s, node.children[idx], encoder)

        #undomove!(s, undo)

    end
end


function backpropagate(r, node)
    while node.idx != 0
        # update count and total reward
        idx = node.idx
        node.parent.cN[idx] += 1
        node.parent.cW[idx] += -r # if we lost (r = -1), then our parent did well

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
    U = c .* P / (1 .+ N) * √sum(N)
    return argmax(Q + U)
end


function validpolicy(s, p, encoder)
    # list of valid moves, according to chess rules
    a = moves(s)
    nmoves = length(a)

    # extract the policy values for only the valid actions
    # by converting uci moves to alphazero plane moves
    # and then reading the value from the planes
    vp = Vector{Float16}(undef, nmoves)
    for i in 1:nmoves
        row, col, plane = alphazero_rep(a[i], encoder)
        vp[i] = p[row, col, plane]
    end

    return nmoves, a, vp
end





function playgame()
    encoder, decoder = alphazero_encoder_decoder()

    root = TreeNode()
    root.idx = 0

    for _ in 1:100
        s = startboard()
        mcts(s, root, encoder)
    end

    # print_tree(root, maxdepth=20)
end



playgame()

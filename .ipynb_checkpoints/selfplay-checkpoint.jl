# using Pkg
# Pkg.add("BSON")

using Chess         # General chess package, used throughout the project
using BSON: @load, @save
#using BenchmarkTools

include("translation.jl") # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for monte carlo searches
include("neuralnet.jl") # the brain that helps monte carlo focus its search

function mcts(s::Board, node::TreeNode, fθ::Chain, encoder::Dict)
    if isdraw(s) # we drew, so no reward, but still need to update counts to show
        #backpropagate(0, node)
        return 
        
    elseif ischeckmate(s) # prev player checkmated us, negative reward
        #backpropagate(-1, node)
        return 
        
    lock(node) do
         if node.isleaf
            node.isleaf = false # we have now explored it, no longer a leaf
            p, v = fθ(alphazero_rep(s)) # ask neural net for policy and value
            v = v[1]

            nmoves, a, p = validpolicy(s, p, encoder) # extract legal moves

            node.cA, node.cP = a, p # store moves and policy
            node.cN = node.cW = zeros(Float16, nmoves) # no statistics on children yet

            node.children = [TreeNode() for _ in 1:nmoves] 
            bookkeep_children!(node, nmoves::Int) # link children & parents
            #backpropagate(v, node) # update parents with the reward
            return 
        end
    end
        
    else
        if @isdefined node
            lock(node) do
                idx = argmaxUCT(node.cW, node.cN, node.cP) # greedily choose max UCT score
                a = node.cA[idx]  # use the chosen action to transition to new state
                child = node.children[idx]
            end
            domove(!s, a)  # (modify state in place, can use undomove!(s, undo) if needed)
            mcts(s, child, fθ, encoder) # rollout (using neural net)
            return 
        end
        
        
    end
end


function backpropagate(r, node)
    while node.idx != 0
        idx = node.idx
        
        node.parent.cN[idx] += 1
        # if we lost (r = -1), then our parent acted well by choosing us as their child
        node.parent.cW[idx] += -r 
        
        node = node.parent # parent is opposite player,
        r = -r # our win is their loss
    end
end


function argmaxUCT(W, N, P, c=2.0)
    Q = W ./ N
    U = c * P ./ (1 .+ N) * √sum(N)
    return argmax(Q .+ U)
end


function validpolicy(s, p, encoder)
    a = moves(s) # list of valid moves, according to chess rules
    p = reshape(p, (8, 8, 88)) # neural net spits p out as a 1-d vector
    nmoves = length(a)

    vp = Vector{Float16}(undef, nmoves)
    for i in 1:nmoves
        row, col, plane = alphazero_rep(a[i], encoder) # translate uci to az encoding 
        vp[i] = p[row, col, plane] # extract the policy values from nnet output 
    end
    
    vp = vp ./ sum(vp)
    return nmoves, a, vp
end



function playgame(nsims)
    encoder, decoder = alphazero_encoder_decoder()

    root = TreeNode()
    root.idx = 0
    
    m = fθ()
    Threads.@threads for _ in 1:nsims
        s = startboard()
        mcts(s, root, m, encoder)
    end
end


playgame(100)


# function save_model()
#     @save "model.bson" neuralnet
# end

# function load_model()
#     @load "model.bson" neuralnet
# end

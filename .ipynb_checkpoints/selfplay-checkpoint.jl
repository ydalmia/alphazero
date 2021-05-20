# RUN SETTINGS: julia --threads 8 
# may also want: --optimize=3 --math-mode=fast --inline=yes --check-bounds=no

# Further Ideas:
# --------------
# Make Model Deeper?
#
# Batch Image Classification Jobs?
# Cache Fenstrings?
# Try KNet instead of Flux? 
# Change Model Input to Lower Dimension? i.e. original 79 instead of 88

using Chess: isdraw, ischeckmate, Board, domove!, tostring, pprint
using Base.Threads
using CUDA
using Flux
#using BSON: @load, @save # for serializing model

include("translation.jl") # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for monte carlo searches
include("neuralnet.jl") # the brain that helps monte carlo focus its search

const VIRTUAL_LOSS = Float32(-1.0)
const VIRTUAL_WIN = -VIRTUAL_LOSS

const encoder, _ = alphazero_encoder_decoder()


# NOTE: if you want to check how many simulations were actually performed
# then return 1 or 0 from each branch.
function mcts!(s::Board, node::TreeNode)    
    if isdraw(s) # we drew => no reward, still need to update counts
        backpropagate!(0.0, node)
    elseif ischeckmate(s) # prev player checkmated us, we lost
        backpropagate!(-1.0, node)
    elseif atomic_cas!(node.isleaf, true, false) # if leaf, mark not leaf, return old isleaf val.  
        r = expand!(node, s)
        atomic_xchg!(node.isready, true)
        backpropagate!(r, node)
    else
        while atomic_cas!(node.isready, false, false) == false # spin until ready
            sleep(0.1)
        end
        W, N, P = children_stats(node)            # by false in previous elseif, but, 
        child = node.children[argmaxUCT(W, N, P)] # the thread executing expand! is not done
        atomic_add!(child.W, VIRTUAL_LOSS) # virtual loss
        
        domove!(s, child.A)
        mcts!(s, child)
    end
end


function expand!(node::TreeNode, s::Board)      
    base = f(alphazero_rep(s) |> gpu) # ask neural net for policy and value
    p = Array(policy(base))
    v = Array(value(base))
    nmoves, a, vp = validpolicy(s, p, encoder) 
    node.children = [TreeNode(a, vp, node) for (a, vp) in zip(a, vp)]
    return v[1, 1] # heuristic r
end



function backpropagate!(r::Float32, node::TreeNode)
    while node != nothing
        atomic_add!(node.W, r + VIRTUAL_WIN) # reverse virtual loss, and add reward
        atomic_add!(node.N, Float32(1.0))
        node = node.parent
        r = -r
    end
end


function argmaxUCT(W::Vector{Float32}, N::Vector{Float32}, P::Vector{Float32}, c=2.0)
    Q = [N==0.0 ? 0.0 : W/N for (W, N) in zip(W, N)]
    U = c * P ./ (1 .+ N) * √sum(N)
    return argmax(Q .+ U)
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
    return nmoves, a, vp
end


function playgame(nsims::Int)
    root = TreeNode()
    println("Simulating using ", Threads.nthreads(), " threads.")
    @threads for _ in 1:nsims
        mcts!(startboard(), root)    
    end
    println("Explored ", root.N, " positions")  
end

# RUN SETTINGS: julia --threads 8 --optimize=3 --math-mode=fast --inline=yes --check-bounds=no alphazero/selfplay.jl
# ^^ these are for maximum performance, but while debugging should not be used
# ^^ still need to figure out optimal thread setting

# Further Ideas:
# --------------
# Make Model Deeper?
#
# Batch Image Classification Jobs?
# Cache Fenstrings?
# Classify Images w/ GPU? 
# Try KNet instead of Flux? 
# Change Model Input to Lower Dimension? i.e. original 79 instead of 88
# Change for loop in simulation to while? because threading will not do full amount specified

using Chess: isdraw, ischeckmate, Board, domove!, tostring, pprint
using Base.Threads
using BSON: @load, @save # for serializing model
using BenchmarkTools: @btime

include("translation.jl") # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for monte carlo searches
include("neuralnet.jl") # the brain that helps monte carlo focus its search


const VIRTUAL_LOSS = Float32(-1.0)
const VIRTUAL_WIN = -VIRTUAL_LOSS

const encoder, _ = alphazero_encoder_decoder()
const m = fθ()

# NOTE: if you want to check how many simulations were actually performed
# then return 1 or 0 from each branch.
function mcts!(s::Board, node::TreeNode)    
    if isdraw(s) # we drew => no reward, still need to update counts
        backpropagate!(0.0, node)
    
    elseif ischeckmate(s) # prev player checkmated us, we lost
        backpropagate!(-1.0, node)

    elseif atomic_xchg!(node.isleaf, false) # if leaf, mark not leaf, return true. 
        r = nothing                         # if not leaf, mark not leaf, return false. 
        with_lock(node) do 
            r = expand!(node, s)
        end
        backpropagate!(r, node)
            
    elseif node.children != nothing               # mandatory b/c thread may get bumped off
        W, N, P = children_stats(node)            # by false in previous elseif, but, 
        child = node.children[argmaxUCT(W, N, P)] # the thread executing expand! is not done
        
        atomic_add!(child.W, VIRTUAL_LOSS) # virtual loss
        
        domove!(s, child.A)
        mcts!(s, child)
    end
end


function expand!(node::TreeNode, s::Board)            
    p, v = m(alphazero_rep(s)) # ask neural net for policy and value

    nmoves, a, vp = validpolicy(s, p, encoder) 
    node.children = [TreeNode(a, vp, node) for (a, vp) in zip(a, vp)]
    
    return v[1] # heuristic r
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


# Note: add an atomic counter to actually run the expected nsims
# (or, we could change the mcts code to wait for unlock?)
function playgame(nsims::Int)
    root = TreeNode()
    @threads for _ in 1:nsims
        s = startboard()
        mcts!(s, root)
    end
end

@btime playgame(5000)
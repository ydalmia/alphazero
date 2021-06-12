using Base: Unicode
# RUN SETTINGS: julia --threads 8
# may also want: --optimize=3 --math-mode=fast --inline=yes --check-bounds=no
using Chess: isdraw, ischeckmate, Board, domove!, tostring, pprint
using Base.Threads
using Flux
using StatsBase

#using BSON: @load, @save # for serializing model

include("translation.jl") # translate AlphaZero <-> Universal Chess Interface
include("montecarlo_tree.jl") # a tree structure for monte carlo searches
include("neuralnet.jl") # the brain that helps monte carlo focus its search

# const VIRTUAL_LOSS = Float32(-1.0)
# const VIRTUAL_WIN = -VIRTUAL_LOSS

const VIRTUAL_LOSS = Float32(0.0)
const VIRTUAL_WIN = -VIRTUAL_LOSS

const encoder, _ = alphazero_encoder_decoder()


# NOTE: if you want to check how many simulations were actually performed
# then return 1 or 0 from each branch.
function mcts(s::Board, node::TreeNode)
    if isdraw(s) # we drew => no reward, still need to update counts
        backpropagate!(Float32(0.0), node)
        println("THE GAME WAS A DRAW")
    elseif ischeckmate(s) # prev player checkmated us, we lost
        backpropagate!(Float32(1.0), node)
        println("THE GAME WAS A CHECKMATE")
    elseif atomic_cas!(node.isleaf, true, false) # if leaf, mark not leaf, return old isleaf val.
        # println("EXPAND")
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

        sp = domove(s, child.A)
        sp = flip(flop(sp))
        mcts(sp, child)
    end
end


function expand!(node::TreeNode, s::Board)
    base = f(alphazero_rep(s)) # ask neural net for policy and value
    p = Array(policy(base))
    v = Array(value(base))
    if v[1,1] > 1
        println("V > 1")
    end
    a, vp = validpolicy(s, p, encoder)

    a = map(a) do x
        if !ispromotion(x) && ptype(pieceon(s, from(x))) == PAWN && rank(from(x)) == SS_RANK_7 #actually a queen promotion
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
    while node != nothing
        atomic_add!(node.W, r + VIRTUAL_WIN) # reverse virtual loss, and add reward
        atomic_add!(node.N, Float32(1.0))
        node = node.parent
        r = -r
    end
end


function argmaxUCT(W::Vector{Float32}, N::Vector{Float32}, P::Vector{Float32}, c=200.0)
    Q = [N==0.0 ? 0.0 : W/N for (W, N) in zip(W, N)]
    U = c * 1 ./ (1 .+ N) * √sum(N)
    return argmax(our_softmax.(Q .+ U, 0.99))
end

function our_softmax(val, bound)
    if val > bound
        val = bound + (val - bound)/10000000
    end
    return val
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
    @threads for _ in 1:nsims
        mcts(s, root)
    end
end

function playgame(s=startboard())
    tree = TreeNode()
    history = [] # state, π, z

    # Break when game is over
    # while(false)
        if isdraw(s)
            z = Float32(0)
            # break
        elseif ischeckmate(s)
            z = Float32(-1)
            # break
        end

        # simulate what moves are best
        simulate(tree, s, 500)

        # choose move from weighted probability distribution
        _, N, _ = children_stats(tree)
        idx = 1:length(N)
        weights = N / sum(N) # =[0.1, 0.1, 0.2, 0.2, 0.1, 0.3]
        idx = sample(idx, ProbabilityWeights(weights))

        print_chess_tree(tree, 0)

        # prune the tree (tree = tree's selected child)
        tree = tree.children[idx]
        tree.parent = nothing

        append!(history, [s, weights])

        # make move flip board
        s = domove(s, tree.A)
        s = flip(flop(s))

        pprint(s, unicode=true, color=true)
    # end

    # for i in length(history):1:-1
    #     append!(history[i], z)
    #     z = -z
    # end

    # return history
end

playgame(fromfen("7k/4KRpp/4PP2/8/8/8/8/8 w - - 0 1"))

# function train(ntrain::Int)
#     examples = []
#     for i in 1:ntrain
#         append!(examples, playgame())
#     end
# end

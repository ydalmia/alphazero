using AbstractTrees

import Chess.Move



import Base.lock
import Base.ReentrantLock
using Base.Threads

mutable struct TreeNode
    isleaf::Atomic{Bool} # at the bottom of the mcts search tree?

    W::Atomic{Float32} # store the children's Weight
    N::Atomic{Float32} # ... Count

    A::Union{Move, Nothing}    # UCI move to get here
    P::Float32 # ... Policy
    
    parent::Union{TreeNode, Nothing} # store reference to parent
    children::Union{Vector{TreeNode}, Nothing} # store reference to children

    lock::ReentrantLock # local locking of the node
    TreeNode() = new(
        Atomic{Bool}(true), 
        Atomic{Float32}(0.0), 
        Atomic{Float32}(0.0), 
        nothing, 
        0.0, 
        nothing, 
        nothing, 
        ReentrantLock()
    )
    
    TreeNode(a::Move, p::Float32,  parent::TreeNode) = new(
        Atomic{Bool}(true), 
        Atomic{Float32}(0.0), 
        Atomic{Float32}(0.0), 
        a, 
        p, 
        parent, 
        nothing, 
        ReentrantLock()
    )
end


# convenience wrapper that takes care of locking and unlocking
# if you use it as follows:
# with_lock(node) do
#     ... 
#     ...
# end
with_lock(f, node::TreeNode) = lock(f, node.lock)


function children_stats(node::TreeNode)
    n = length(node.children)
    W = Array{Float32}(undef, n)
    N = Array{Float32}(undef, n)
    P = Array{Float32}(undef, n)

    # see https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.atomic_cas!
    for (i, child) in enumerate(node.children)
        W[i] = atomic_cas!(child.W, Float32(0.0), Float32(0.0)) # returns child.W without getting corrupted
        N[i] = atomic_cas!(child.N, Float32(0.0), Float32(0.0)) # ... N
        P[i] = child.P
    end
    return W, N, P
end


# NOT THREAD SAFE, MEANT FOR PRETTY PRINTING RESULTS ONCE DONE
function AbstractTrees.children(node::TreeNode)
    if node.children != nothing
        return [child for child in node.children if child.N > 0.1]
    else
        return ()
    end
end
    
AbstractTrees.printnode(io::IO, node::TreeNode) = (
    print(io, "N: ", node.N)
)

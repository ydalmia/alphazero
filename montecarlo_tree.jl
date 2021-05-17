using Pkg
Pkg.add("AbstractTrees")
using AbstractTrees

# MCTS Tree
# broadly, each node in the tree stores the statistics of the children,
# to improve spatial locality. For example, when calculating UCT scores,
# we need to access each child's statistics, which means 30+ pointer chases.
# Arrays avoid pointer chasing and allow vectorized UCT calculations
mutable struct TreeNode
    isleaf::Bool # at the bottom of the mcts search tree?

    cW::Vector{Float16} # store the children's Weight
    cN::Vector{Float16} # ... Count
    cP::Vector{Float16} # ... Policy

    cA::Vector{Move} # UCI move that will transition board to child state

    children::Vector{TreeNode} # store reference to children

    parent::TreeNode # store reference to parent

    idx::UInt16 # store what index we are in our parent array

    TreeNode() = new(true) # default constructor for self referential objects

end


# helper functions for pretty printing tree
function AbstractTrees.children(node::TreeNode)
    if isdefined(node, :children) # yes children
        return node.children
    end
    return () # no children
end

AbstractTrees.printnode(io::IO, node::TreeNode) = print(io, node.idx)

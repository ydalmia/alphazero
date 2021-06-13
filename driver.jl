using Distributed
# if you want to use all the cores on your computer, 
# uncomment addprocs, everywhere, and distributed lines. 
# its approx-linear speed up in the number of cores you have
# (this may be different with gpu)

# addprocs(7) # adjust to the number of cores you have

# @everywhere 
include("mcts.jl")

function playgames(s, ngames=1; nsims=1000)
    # @distributed 
    for _ in 1:ngames
        root = TreeNode()
        simulate(root, s, nsims)
        println("done")
        print_tree(root, maxdepth=2)
    end
end

@time fetch(playgames(fromfen("7k/4KRpp/4PP2/8/8/8/8/8 w - - 0 1")))
@time fetch(playgames(fromfen("8/1p1k1p1b/5PBr/8/2r5/1p2P1Q1/1KppR2R/8 w - - 0 1")))
@time fetch(playgames(fromfen("7k/4KRpp/4PP2/8/8/8/8/8 w - - 0 1")))
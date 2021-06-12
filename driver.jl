using Distributed

addprocs(4)

@everywhere include("selfplay.jl")

function parallel_playgame(s; nsims=800)
    @distributed for _ in 1:12
        root = TreeNode()
        simulate(root, s, nsims)
        println("done")
        # print_tree(root, maxdepth=2)
    end
end

# 3x Speedup using 4 Cores, 15.0s to 5.6s
@time fetch(playgame(fromfen("7k/4KRpp/4PP2/8/8/8/8/8 w - - 0 1")))
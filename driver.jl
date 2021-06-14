using Distributed

# if you want to use all the cores on your computer, 
# uncomment addprocs, everywhere, and distributed lines. 
# its approx-linear speed up in the number of cores you have
# (this may be different with gpu)

addprocs(6) # adjust to the number of cores you have

@everywhere using StatsBase
@everywhere using JLD2
@everywhere include("mcts.jl")

function playgames(s=startboard()::Board; ngames=6, nsims=800)
    # all_games_data = SharedArray{any}()
    @distributed for _ in 1:ngames
        root = TreeNode()
        wholeTree = root
        train_x = []

        z = nothing
        while(true)
            pprint(s, unicode=true, color=true)
            println("move simulated")
            if ischeckmate(s) 
                z = 1 # lol vivek check this
                break
            elseif isdraw(s)
                z = 0
                break
            else
                # mcts
                simulate(root, s, nsims)
                _, N, _ = children_stats(root)
                π = N / sum(N)

                # save example to train from later
                push!(train_x, (alphazero_rep(s), π))

                # randomly sample child state according to π
                idx = sample(1:length(N), ProbabilityWeights(π))
                root = root.children[idx]
                s = domove(s, root.A)
            end
        end

        for i in length(train_x):1:-1
            train_x[i] = (train_x[i]..., z)
            z = -z
        end

        # println(train_x)
        print_tree(wholeTree, maxdepth=3)
        core=myid()
        # Serialization.serialize(open("games_data_core_$core.json", "w"), train_x)
        @save "games_data_core_$core.jld2" train_x

    end
end

# @time fetch(playgames(fromfen("8/1p1k1p1b/5PBr/8/2r5/1p2P1Q1/1KppR2R/8 w - - 0 1")))
# @time fetch(playgames(fromfen("7k/4KRpp/4PP2/8/8/8/8/8 w - - 0 1"), ngames=1)) # easy 1 move to checkmate
# @time fetch(playgames(fromfen("kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1"), ngames=1, nsims=2000)) # 2 move to checkmate
@time playgames()
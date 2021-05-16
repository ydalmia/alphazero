# https://fluxml.ai/Flux.jl/stable/gpu/

using Pkg
Pkg.add("Flux")
using Flux


function Residual()
    α = Chain(Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
                BatchNorm(256, relu),
                Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
                BatchNorm(256))

    β = SkipConnection(α, +)
    return β
end

function fθ(x)
    m = Chain(
            Conv((3, 3), 3 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256, relu),
            Residual(),
            Residual()
        )

    p = Chain(
        m,
        Conv((1, 1), 256 => 2, relu, stride=(1, 1)),
        BatchNorm(2, relu),
        flatten,
        Dense(8*8*2, 8*8*88))
    )

    # v = Chain(
    #     m,
    #     Conv((1, 1), 256 => 1, relu, stride=(1, 1))
    #     flatten,
    #     Dense(256, 256),
    #     tanh()
    # )
    return p
    # , v
end


# FIGURE OUT HOW TO CONVERT TO FLOAT16 FOR EFFICIENCY
x = rand(Float32, 8, 8, 3, 1)

fθ(x)



# AlphaLayer
function AlphaLayer(x)
    probvec = normalize(exp(x[1:length(x)-1]), 1)
    val = 2*sigmoid(last(x)) - 1


    return [probvec, val]
end

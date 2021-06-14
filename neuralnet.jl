using Flux
# using CUDA

Residual() = SkipConnection(
    Chain(Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256, relu),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256)
    ), 
+)

f = Chain(
    Conv((3, 3), 17 => 256, relu, pad=(1, 1), stride=(1, 1)), # 17 feature planes
    BatchNorm(256, relu),
    Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
    BatchNorm(256),
    Residual(),
    Residual(),
    Residual(),
    Residual(), 
    Residual(),
    Residual()
)
# ) |> gpu

policy = Chain(
    Conv((1, 1), 256 => 2, relu, stride=(1, 1)),
    BatchNorm(2, relu),
    flatten,
    Dense(8*8*2, 8*8*73, sigmoid) # 73 types of moves, 8*8 possible src squares
)
# ) |> gpu

value = Chain(
    Conv((1, 1), 256 => 1, relu, stride=(1, 1)),
    BatchNorm(1, relu),
    flatten,
    Dense(8*8*1, 256, relu),
    Dense(256, 1, tanh) 
)
# ) |> gpu
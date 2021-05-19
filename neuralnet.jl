using Flux

function save_model(fθ)
    @save "model.bson" fθ
end

function load_model()
    return @load "model.bson"
end


function fθ()
    Chain(
        Conv((3, 3), 13 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256, relu),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Residual(),
        Residual(),
        Residual(),
        Residual(),
        Residual(),
        Split(
            p(), 
            v()
        )  
    )
end


function Residual()
    SkipConnection(
        Chain(Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256, relu),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256)
        ), 
    +)
end


function p()
    Chain(
        Conv((1, 1), 256 => 2, relu, stride=(1, 1)),
        BatchNorm(2, relu),
        flatten,
        Dense(8*8*2, 8*8*88),
    )
end

function v()
    Chain(
        Conv((1, 1), 256 => 1, relu, stride=(1, 1)),
        BatchNorm(1, relu),
        flatten,
        Dense(8*8*1, 256, relu),
        Dense(256, 1),
        tanh
    )
end


# To allow a network to have multiple "heads"
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

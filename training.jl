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


function fθ()
    γ = Chain(
            Conv((3, 3), 13 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256, relu),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Residual(),
            Residual(),
            Residual(),
            Residual()
    )
    return γ
end


x = rand(Float32, (8, 8, 119, 5))
y = fθ(x)

print(size(y))

function l(y, ŷ)
    value_loss = (y.z - ŷ.v)^2
    policy_loss = -y.π ⋅ log(ŷ.ρ)
    return value_loss + policy_loss
end


Flux.train!(objective, params, data, opt)
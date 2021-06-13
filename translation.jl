using Chess

ENCODING_TYPE = Tuple{SquareDelta, Union{PieceType, Nothing}}
function generate_encoder_decoder()
    move_list = Vector{ENCODING_TYPE}()
    encoder = Dict{ENCODING_TYPE, Int64}()
    decoder = Dict{Int64, ENCODING_TYPE}()

    # queen moves
    for n in 1:7
        for delta in [DELTA_N, DELTA_S, DELTA_E, DELTA_W, DELTA_NW, DELTA_NE, DELTA_SW, DELTA_SE]
            push!(move_list, (n * delta, nothing))
        end
    end
    # knight moves
    for delta_file in [2 * DELTA_E, 2 * DELTA_W]
        for delta_rank in [DELTA_N, DELTA_S]
            push!(move_list, (delta_file + delta_rank, nothing))
        end
    end
    # more knight moves (4)
    for delta_file in [DELTA_E, DELTA_W]
        for delta_rank in [2 * DELTA_N, 2 * DELTA_S]
            push!(move_list, (delta_file + delta_rank, nothing))
        end
    end
    # promotion moves (12)
    for delta in [DELTA_N, DELTA_NE, DELTA_NW]
        for piece in [KNIGHT, BISHOP, ROOK]
            push!(move_list, (delta, piece))
        end
    end
    # encoder, decoder
    for (i, mv) in enumerate(move_list)
        encoder[mv] = i
        decoder[i] = mv
    end

    return encoder, decoder
end
const encoder, decoder = generate_encoder_decoder()

function encode(a, side::PieceColor)
    encoded_a = map(a) do x
        delta = to(x) - from(x)
        src = from(x)

        if side == BLACK 
            src = Square(65 - src.val) # rotate 180 degrees
            delta = -1 * delta # opposite vector in mathematical sense
        end

        # elide queen promotions and pawns moving into the final rank of the board
        # so that we only explicitly encode under promotions. ie, a queen promotion
        # will have prmt = nothing
        prmt = nothing
        if ispromotion(x) && promotion(x) != QUEEN
            prmt = promotion(x)
        end

        mv = (delta, prmt)::ENCODING_TYPE
        return src, mv
    end
    return encoded_a
end

function valid_policy(s, p)
    a = moves(s)
    encoded_a = encode(a, sidetomove(s))

    p = reshape(p, 64, 73)
    vp = [p[src.val, encoder[mv]] for (src, mv) in encoded_a]
    return a, vp
end

function get_layers(b, side) 
    side_pieces = nothing
    if side == WHITE
        side_pieces = [PIECE_WP, PIECE_WN, PIECE_WB, PIECE_WR, PIECE_WQ, PIECE_WK]
    else 
        side_pieces = [PIECE_BP, PIECE_BN, PIECE_BB, PIECE_BR, PIECE_BQ, PIECE_BK]
    end

    piece_layers = [toarray(pieces(b, piece)) for piece in side_pieces]
    king_castle = cancastlekingside(b, side) ? ones(8, 8) : zeros(8, 8)
    queen_castle = cancastlequeenside(b, side) ? ones(8, 8) : zeros(8, 8)

    layers = cat(king_castle, queen_castle, piece_layers..., dims=3)
    return layers
end

# util function to rotate each 8x8 layer in layers
rotate_layers(layers) = mapslices(rot180, layers, dims=[1, 2]) 

function alphazero_rep(b::Board)
     # TODO: apply the rotate board function below
     # TODO: should we always have black on top of white? or current player on bottom?
    layers = cat(
        get_layers(b, WHITE),
        rotate_layers(get_layers(b, BLACK)),
        sidetomove(b) == WHITE ? ones(8, 8) : zeros(8, 8), 
        dims=3
    )
    return convert(Array{Float32}, layers) # cast to float32 for faster ML
end


# black's pawn layer, it should have the pawns towards
# the bottom of the screen if rotate works as intended
# alphazero_rep(startboard())[:, :, 9]
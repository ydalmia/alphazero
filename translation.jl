using Chess

ENCODING_TYPE = Tuple{SquareDelta, Union{PieceType, Nothing}}
function generate_encoder_decoder()
    # queen moves
    queen_deltas = [DELTA_N, DELTA_S, DELTA_E, DELTA_W, DELTA_NW, DELTA_NE, DELTA_SW, DELTA_SE]
    queen_moves = [((n * d, nothing) for n in 1:7, d in queen_deltas)...] # flatten matrix by splatting
    
    # knight moves
    delta_files = [DELTA_E, DELTA_W]
    delta_ranks = [DELTA_N, DELTA_S]
    knight_moves1 = [((df + dr, nothing) for df in 2*delta_files, dr in delta_ranks)...]
    knight_moves2 = [((df + dr, nothing) for df in delta_files, dr in 2*delta_ranks)...]

    # under-promotion moves
    pawn_prmt_delta = [DELTA_N, DELTA_NE, DELTA_NW]
    prmt_pieces = [KNIGHT, BISHOP, ROOK]
    prmt_moves = [((d, piece) for d in pawn_prmt_delta, piece in prmt_pieces)...]
    
    move_list = cat(queen_moves, knight_moves1, knight_moves2, prmt_moves, dims=1)
    encoder = Dict{ENCODING_TYPE, Int64}()
    decoder = Dict{Int64, ENCODING_TYPE}()

    # encoder, decoder
    for (i, mv) in enumerate(move_list)
        encoder[mv] = i
        decoder[i] = mv
    end

    return encoder, decoder
end

const encoder, decoder = generate_encoder_decoder()

function encode(a::MoveList, side::PieceColor)
    encoded_a = map(a) do x
        delta = to(x) - from(x)
        src = from(x)

        if side == BLACK 
            src = Square(65 - src.val) # rotate 180 degrees
            delta = -1 * delta # opposite vector in mathematical sense
        end

        # elide queen promotions and pawns moving into the final rank of the board
        # so that we only explicitly encode underpromotions. ie, a queen promotion
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

function valid_policy(s::Board, p::Array{Float32})
    a = moves(s)
    encoded_a = encode(a, sidetomove(s))

    p = reshape(p, 64, 73)
    vp = [p[src.val, encoder[mv]] for (src, mv) in encoded_a]
    return a, vp
end

function get_layers(b::Board, side::PieceColor)::Array{Float32} 
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
rot180(layers::Array{Float32, 3}) = mapslices(Base.rot180, layers, dims=[1, 2]) 

function alphazero_rep(b::Board)
     # TODO: apply the rotate board function below
     # TODO: should we always have black on top of white? or current player on bottom?
    layers = nothing
    if sidetomove(b) == WHITE
        layers = cat(
            get_layers(b, WHITE),
            get_layers(b, BLACK),
            ones(8, 8), 
            dims=3)
    else 
        layers = cat(
            rot180(get_layers(b, BLACK)),
            rot180(get_layers(b, WHITE)),
            zeros(8, 8), 
            dims=3) 
    end
    return convert(Array{Float32}, layers) # cast to float32 for faster ML
end
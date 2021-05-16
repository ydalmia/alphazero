using Pkg
Pkg.add("Chess")
using Chess

# simple representation of board: just pieces plus current color
# for a total of 13 planes
function get_alphazero_input(board, rep)
    color = sidetomove(board) == WHITE ? ones(rep, 8, 8) : zeros(rep, 8, 8)

    input = cat(
        color,
        toarray(pieces(board, PIECE_BP), rep),
        toarray(pieces(board, PIECE_BN), rep),
        toarray(pieces(board, PIECE_BB), rep),
        toarray(pieces(board, PIECE_BR), rep),
        toarray(pieces(board, PIECE_BQ), rep),
        toarray(pieces(board, PIECE_BK), rep),
        toarray(pieces(board, PIECE_WP), rep),
        toarray(pieces(board, PIECE_WN), rep),
        toarray(pieces(board, PIECE_WB), rep),
        toarray(pieces(board, PIECE_WR), rep),
        toarray(pieces(board, PIECE_WQ), rep),
        toarray(pieces(board, PIECE_WK), rep),
        dims=3
    )
    return input
end


function alphazero_rep(move::Move, encoder::Dict)
    dest = to(move)
    src = from(move)

    src_row = rank(src)
    src_col = file(src)

    delta = dest - src
    prmt = ispromotion(move) ? promotion(move) : EMPTY
    plane = encoder[delta, prmt]

    return (src_row, src_col, plane)
end

function uci_rep(src_row::Int, src_col::Int, plane::Int, decoder::Dict)
    # row 8 in matrix -> rank 1
    # col 8 in matrix -> rank H
    sq = Square(SquareFile(src_col), SquareRank(src_row))

    delta, prmt = decoder[plane]

    if prmt == EMPTY
        return Move(sq, sq + delta)
    end

    return Move(sq, sq + delta, prmt)
end


function alphazero_encoder_decoder()
    encoder = Dict()
    decoder = Dict()

    i = 1

    # queen moves (56)
    for n in 1:7
        for delta in [DELTA_N, DELTA_S, DELTA_E, DELTA_W, DELTA_NW, DELTA_NE, DELTA_SW, DELTA_SE]
            encoder[(n * delta, EMPTY)] = i
            decoder[i] = (n * delta, EMPTY)
            i += 1
        end
    end

    # knight moves (4)
    for delta_file in [2 * DELTA_E, 2 * DELTA_W]
        for delta_rank in [DELTA_N, DELTA_S]
            delta = delta_file + delta_rank
            encoder[(delta, EMPTY)] = i
            decoder[i] = (delta, EMPTY)
            i += 1
        end
    end

    # more knight moves (4)
    for delta_file in [DELTA_E, DELTA_W]
        for delta_rank in [2 * DELTA_N, 2 * DELTA_S]
            delta = delta_file + delta_rank
            encoder[(delta, EMPTY)] = i
            decoder[i] = (delta, EMPTY)
            i += 1
        end
    end

    # promotion moves (24)
    for delta in [DELTA_N, DELTA_NE, DELTA_NW, DELTA_S, DELTA_SE, DELTA_SW]
        for piece in [KNIGHT, BISHOP, ROOK, QUEEN]
            encoder[(delta, piece)] = i
            decoder[i] = (delta, piece)
            i += 1
        end
    end

    return encoder, decoder
end


# to and from uci and alphazero
encoder, decoder = alphazero_encoder_decoder()

# test all values in decoder (and therefore encoder, in theory)
for i in 1:88
    decoder[i]
end

# test translation from uci -> alpha zero
println(alphazero_rep(Move(SQ_A7, SQ_B8, KNIGHT), encoder))
println(alphazero_rep(Move(SQ_A7, SQ_A8, QUEEN), encoder))
println(alphazero_rep(Move(SQ_A4, SQ_C6), encoder))
println(alphazero_rep(Move(SQ_H8, SQ_A1), encoder))
println(alphazero_rep(Move(SQ_B2, SQ_A1), encoder))
# test translation from alpha zero -> uci
println(uci_rep(1, 3, 79, decoder))

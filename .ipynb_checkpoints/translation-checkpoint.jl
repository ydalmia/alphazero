# using Pkg
# Pkg.add("Chess")
using Chess


# simple representation of board: just pieces plus current color
# for a total of 13 planes
function alphazero_rep(board::Board)
    board_rep = Array{Float32, 4}(undef, 8, 8, 13, 1)
    
    board_pieces = [PIECE_BP, PIECE_BN, PIECE_BB, PIECE_BR, 
                PIECE_BQ, PIECE_BK, PIECE_WP, PIECE_WN, 
                PIECE_WB, PIECE_WR, PIECE_WQ, PIECE_WK]
    
    for (i, piece) in enumerate(board_pieces)
        board_rep[:, :, i, 1] = toarray(pieces(board, piece))
    end
    
    color = sidetomove(board) == WHITE ? ones(Float32, 8, 8) : zeros(Float32, 8, 8)
    board_rep[:, :, 13, 1] = color
    return board_rep
end


function alphazero_rep(move::Move, encoder::Dict)
    dest = to(move)
    src = from(move)

    src_row = rank(src).val
    src_col = file(src).val

    delta = dest - src
    prmt = ispromotion(move) ? promotion(move) : EMPTY
    plane = encoder[delta, prmt]

    return (src_row, src_col, plane)
end


function uci_rep(src_row::Int, src_col::Int, plane::Int, decoder::Dict)
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
# encoder, decoder = alphazero_encoder_decoder()

# # test all values in decoder (and therefore encoder, in theory)
# for i in 1:88
#     decoder[i]
# end
#
# # test translation from uci -> alpha zero
# println(alphazero_rep(Move(SQ_A7, SQ_B8, KNIGHT), encoder))
# println(alphazero_rep(Move(SQ_A7, SQ_A8, QUEEN), encoder))
# println(alphazero_rep(Move(SQ_A4, SQ_C6), encoder))
# println(alphazero_rep(Move(SQ_H8, SQ_A1), encoder))
# println(alphazero_rep(Move(SQ_B2, SQ_A1), encoder))
# # test translation from alpha zero -> uci
# println(uci_rep(1, 3, 79, decoder))

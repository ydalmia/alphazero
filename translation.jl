using Chess

function alphazero_rotate(old_move::Move)
    new_src = Square(65 - from(old_move).val)
    new_dest = Square(65 - to(old_move).val)

    if ispromotion(old_move)
        return Move(new_src, new_dest, promotion(old_move))
    else
        return Move(new_src, new_dest)
    end
end

function alphazero_rep(board::Board)
    board_rep = Array{Float32}(undef, 8, 8, 18)

    board_pieces = [PIECE_WP, PIECE_WN, PIECE_WB, PIECE_WR,
                    PIECE_WQ, PIECE_WK, PIECE_BP, PIECE_BN,
                    PIECE_BB, PIECE_BR, PIECE_BQ, PIECE_BK]

    BLACK_KING_CASTLE = cancastlekingside(board, BLACK)
    WHITE_KING_CASTLE = cancastlekingside(board, WHITE)

    BLACK_QUEEN_CASTLE = cancastlequeenside(board, BLACK)
    WHITE_QUEEN_CASTLE = cancastlequeenside(board, WHITE)

    CURR_COLOR = sidetomove(board)

    perspectiveCorrectedBoard = (CURR_COLOR == BLACK) ? rotate(board) : board

    CURRENT_PERSPECTIVE_KING_CASTLE = (CURR_COLOR == BLACK) ? BLACK_KING_CASTLE : WHITE_KING_CASTLE
    CURRENT_PERSPECTIVE_QUEEN_CASTLE = (CURR_COLOR == BLACK) ? BLACK_QUEEN_CASTLE : WHITE_QUEEN_CASTLE

    OPPOSITE_PERSPECTIVE_KING_CASTLE = (CURR_COLOR == BLACK) ? WHITE_KING_CASTLE : BLACK_KING_CASTLE
    OPPOSITE_PERSPECTIVE_QUEEN_CASTLE = (CURR_COLOR == BLACK) ? WHITE_QUEEN_CASTLE : BLACK_QUEEN_CASTLE

    for (i, piece) in enumerate(board_pieces)
        board_rep[:, :, i] = toarray(pieces(perspectiveCorrectedBoard, piece))
    end

    board_rep[:, :, 13] .= (CURR_COLOR == WHITE)
    board_rep[:, :, 14] .= CURRENT_PERSPECTIVE_KING_CASTLE
    board_rep[:, :, 15] .= CURRENT_PERSPECTIVE_QUEEN_CASTLE
    board_rep[:, :, 16] .= OPPOSITE_PERSPECTIVE_KING_CASTLE
    board_rep[:, :, 17] .= OPPOSITE_PERSPECTIVE_QUEEN_CASTLE
    board_rep[:, :, 18] .= board.r50

    return board_rep
end

function alphazero_rep(move::Move, encoder::Dict)
    dest = to(move)
    src = from(move)

    src_row = rank(src).val
    src_col = file(src).val

    delta = dest - src
    prmt = EMPTY

    if ispromotion(move)
        prmt = promotion(move)
        if prmt == QUEEN
            prmt = EMPTY
        end
    end

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

    # promotion moves (12)
    for delta in [DELTA_N, DELTA_NE, DELTA_NW]
        for piece in [KNIGHT, BISHOP, ROOK, QUEEN]
            encoder[(delta, piece)] = i
            decoder[i] = (delta, piece)
            i += 1
        end
    end

    return encoder, decoder
end

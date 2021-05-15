    function col_to_int(c::Char)
        return ((c - 'a') + 1)
    end
    
    function row_col_promotion(move_str)
        from_col = letter_to_int(move_str[1])
        from_row = parse(Int, move_str[2])
        
        to_col = letter_to_int(move_str[3])
        to_row = parse(Int, move_str[4])
         
        promotion = NO_PROMOTION
        if (length(move_str) == 5)
            piece = move_str[5]
            if (piece == KNIGHT || piece == BISHOP || piece == ROOK)
                promotion = piece
            end
        end
        
        return from_col, from_row, to_col, to_row, promotion 
    end
    
    function alphazero_encoding(move_str)
        from_col, from_row, to_col, to_row, promotion = row_col_promotion(move_str)
        plane = encoding[(to_col - from_col, to_row - from_row, promotion)]
        return from_col, from_row, plane
    end
   ]
  },
  {
   cell_type: code,
   execution_count: 76,
   metadata: {},
   outputs: [
    {
     name: stdout,
     output_type: stream,
     text: [
      (1, 7, 73)
      (1, 7, 70)
      (1, 4, 51)
      (8, 8, 7)
      (2, 2, 1)\n
     ]
    }
   ],
   source: [
    # some tests for move encoding
    
    move_str = Chess.tostring(Chess.Move(Chess.SQ_A7, Chess.SQ_B8, Chess.KNIGHT))
    println(alphazero_encoding(move_str))
    
    move_str = Chess.tostring(Chess.Move(Chess.SQ_A7, Chess.SQ_A8, Chess.KNIGHT))
    println(alphazero_encoding(move_str))
    
    move_str = Chess.tostring(Chess.Move(Chess.SQ_A4, Chess.SQ_C6))
    println(alphazero_encoding(move_str))
    
    move_str = Chess.tostring(Chess.Move(Chess.SQ_H8, Chess.SQ_A1))
    println(alphazero_encoding(move_str))
    
    move_str = Chess.tostring(Chess.Move(Chess.SQ_B2, Chess.SQ_A1))
    println(alphazero_encoding(move_str))
   ]




function make_encoding()
        encoding = Dict{Int8 Int8 Char) Int8}()
        i = 1
        # queen-type moves
        for up in [-1 0 1] # can move down zero or up
            for right in [-1 0 1] # can move left right or zero
                for n in 1:7 # scalar multiplier for how far we move
                    if up != 0 || right != 0 # need to move somewhere
                        encoding[(n * right n * up NO_PROMOTION)] = i
                        i += 1
                    end
                end
            end
        end
    
        # knight moves
        for up in [-2 2 -1 1] # down or up and how far
            for right in [-2 2 -1 1] # left or right and how far
                if abs(up) != abs(right) # left and right can't be same length
                    encoding[(right up NO_PROMOTION)] = i
                    i += 1
                end
            end
        end
    
        # under promotion moves
        for up in [1]
            for right in [-1 0 1]
                for piece in [ROOK BISHOP KNIGHT]
                    encoding[(right up piece)] = i
                    i += 1
                end
            end
        end
        
        
        return encoding i
    end

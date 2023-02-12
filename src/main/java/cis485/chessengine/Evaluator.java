package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Piece;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.Square;

public class Evaluator {
    private static int[] pieceValues = {1, 3, 3, 5, 9, 0};
    public static int materialBalance(Board board, Side side) {
        int value = 0;
        for (int i = 0; i < 64; i++) {
            Square square = Square.squareAt(i);
            Piece piece = board.getPiece(square);
            if (piece != Piece.NONE) {
                if (piece.getPieceSide() == side) {
                    value += pieceValues[piece.getPieceType().ordinal()];
                }
                else {
                    value -= pieceValues[piece.getPieceType().ordinal()];
                }
            }
        }
        return value;
    }
}

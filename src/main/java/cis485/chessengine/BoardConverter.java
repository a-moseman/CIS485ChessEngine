package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Piece;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.Square;

public class BoardConverter {
    /**
     * todo: test
     * Convert board using one hot encoding.
     * @param board The current board state.
     * @return Board
     */
    public static boolean[][][] convert(Board board) {
        boolean[][][] data = new boolean[8][8][9];
        int x, y;
        for (x = 0; x < 8; x++) {
            for (y = 0; y < 8; y++) {
                Square square = Square.squareAt(x + y * 8); // todo: confirm x + y * 8 is accurate
                Piece piece = board.getPiece(square);
                if (!piece.equals(Piece.NONE)) {
                    Side side = piece.getPieceSide();
                    data[x][y][0] = true;
                    data[x][y][side == Side.WHITE ? 1 : 2] = true;
                    data[x][y][piece.getPieceType().ordinal() + 3] = true;
                }
            }
        }
        return data;
    }
}

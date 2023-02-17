package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BoardConverter {
    /**
     * todo: test
     * Convert board using one hot encoding.
     * @param board The current board state.
     * @return Board
     */
    public static INDArray convert(Board board) {
        INDArray data = Nd4j.create(1, 9, 8, 8); // minibatch, channel, height, width
        int x, y;
        Square square;
        Piece piece;
        Side side;
        for (x = 0; x < 8; x++) {
            for (y = 0; y < 8; y++) {
                square = Square.squareAt(x + y * 8); // todo: confirm x + y * 8 is accurate
                piece = board.getPiece(square);
                if (!piece.equals(Piece.NONE)) {
                    side = piece.getPieceSide();
                    data.putScalar(new int[]{0, 0, x, y}, 1);
                    //data[0][0][x][y] = 1;
                    data.putScalar(new int[]{0, side == Side.WHITE ? 1 : 2, x, y}, 1);
                    //data[0][side == Side.WHITE ? 1 : 2][x][y] = 1;
                    data.putScalar(new int[]{0, piece.getPieceType().ordinal() + 3, x, y}, 1);
                    //data[0][piece.getPieceType().ordinal() + 3][x][y] = 1;
                }
            }
        }
        return data;
    }
}

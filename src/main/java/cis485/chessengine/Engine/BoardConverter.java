package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.*;

public class BoardConverter {
    /**
     * todo: test
     * Convert board using one hot encoding.
     * @param board The current board state.
     * @return Board
     */
    /*
    public static INDArray convert(Board board, boolean forMCTS) {
        INDArray data;
        if (forMCTS) {
            data = Nd4j.create(1, 8, 8, 8); //minibatch, channel, height, width
        }
        else {
            data = Nd4j.create(8, 8, 8); //channel, height, width
        }
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
                    if (forMCTS) {
                        data.putScalar(new int[]{0, side == Side.WHITE ? 0 : 1, x, y}, 1);
                        data.putScalar(new int[]{0, piece.getPieceType().ordinal() + 2, x, y}, 1);
                    }
                    else {
                        data.putScalar(new int[]{side == Side.WHITE ? 0 : 1, x, y}, 1);
                        data.putScalar(new int[]{piece.getPieceType().ordinal() + 2, x, y}, 1);
                    }
                }
            }
        }
        return data;
    }
     */
}

package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BoardConverter {
    /**
     * TODO: explore not using pooling layers like AlphaZero and ConvChess
     * Convert board using one hot encoding.
     * @param board The current board state.
     * @return Board
     */
    public static INDArray convert(Board board, boolean forMCTS) {
        INDArray data;
        if (forMCTS) {
            data = Nd4j.create(1, 12, 8); //minibatch, channel, height, width
        } else {
            data = Nd4j.create(12, 8, 8); //channel, height, width
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
                    int s = side == Side.WHITE ? 0 : 6;
                    int p = piece.getPieceType().ordinal();
                    int i = s + p;
                    if (forMCTS) {
                        data.putScalar(new int[]{0, i, x, y}, 1);
                    } else {
                        data.putScalar(new int[]{i, x, y}, 1);
                    }
                }
            }
        }
        return data;
    }

    public static void main(String[] args) {
        Board boardOne = new Board();
        Board boardTwo = new Board();
        Board boardThree = new Board();
        boardTwo.loadFromFen("8/8/8/4p1K1/2k1P3/8/8/8 b - - 0 1");
        INDArray convOne = BoardConverter.convert(boardOne, false);
        INDArray convTwo = BoardConverter.convert(boardTwo, false);
        INDArray convThree = BoardConverter.convert(boardThree, false);
        System.out.println(convOne.equals(convTwo));
        System.out.println(convOne.equals(convThree));
    }
}

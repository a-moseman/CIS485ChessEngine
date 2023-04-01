package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.*;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class BoardConverter {
    public static float[][][] oneHotEncode(Board board) {
        float[][][] data = new float[12][8][8];
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                Square square = Square.squareAt(x + y * 8);
                Piece piece = board.getPiece(square);
                if (!piece.equals(Piece.NONE)) {
                    Side side = piece.getPieceSide();
                    int i = (side == Side.WHITE ? 0 : 6) + piece.getPieceType().ordinal();
                    data[i][y][x] = 1;
                }
            }
        }
        return data;
    }

    public static DataSet convert(List<float[][][]> features, List<float[]> labels) {
        assert features.size() == labels.size();
        float[][][][] xRaw = new float[features.size()][12][8][8];
        float[][] yRaw = new float[labels.size()][3];
        for (int i = 0; i < features.size(); i++) {
            xRaw[i] = features.get(i);
            yRaw[i] = labels.get(i);
        }
        return new DataSet(Nd4j.create(xRaw), Nd4j.create(yRaw));
    }
}

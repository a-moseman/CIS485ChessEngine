package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.game.Game;
import com.github.bhlangonijr.chesslib.game.GameResult;
import com.github.bhlangonijr.chesslib.move.MoveList;
import com.github.bhlangonijr.chesslib.pgn.PgnIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class SLDataLoader {
    private static final Random RANDOM = new Random();
    private static final int SL_DATA_SIZE_MUL = 300_000;
    private static final int SL_WHITE_WINS = SL_DATA_SIZE_MUL;
    private static final int SL_BLACK_WINS = SL_DATA_SIZE_MUL;
    private static final int SL_TIES = SL_DATA_SIZE_MUL;
    private static final int SL_MIN_ELO = 1200;

    public static DataSet generate() {
        PgnIterator pgnIterator;
        try {
            pgnIterator = new PgnIterator("Z:\\amoseman\\ARCHIVE\\LICHESS\\lichess_db_standard_rated_2023-01.pgn");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Iterator<Game> iterator = pgnIterator.iterator();

        int w = 0;
        int b = 0;
        int t = 0;
        List<float[][][]> features = new ArrayList<>();
        List<float[]> labels = new ArrayList<>();
        while (iterator.hasNext() && !(w >= SL_WHITE_WINS && b >= SL_BLACK_WINS && t >= SL_TIES)) {
            Game game = iterator.next();
            int whiteElo = game.getWhitePlayer().getElo();
            int blackElo = game.getBlackPlayer().getElo();
            int moveCount = game.getHalfMoves().size();
            if (whiteElo < SL_MIN_ELO || blackElo < SL_MIN_ELO || moveCount < 1) {
                continue;
            }
            GameResult gameResult = game.getResult();
            float[] label;
            if (gameResult.toString().equals("WHITE_WON")) {
                if (w >= SL_WHITE_WINS) {
                    continue;
                }
                label = new float[]{1, 0, 0};
                w++;
            }
            else if (gameResult.toString().equals("BLACK_WON")) {
                if (b >= SL_BLACK_WINS) {
                    continue;
                }
                label = new float[]{0, 1, 0};
                b++;
            }
            else {
                if (t >= SL_TIES) {
                    continue;
                }
                label = new float[]{0, 0, 1};
                t++;
            }
            MoveList moveList = game.getHalfMoves();
            String position = moveList.getFen(RANDOM.nextInt(moveList.size()));
            Board board = new Board();
            board.loadFromFen(position);
            float[][][] feature = BoardConverter.oneHotEncode(board);
            features.add(feature);
            labels.add(label);
            if (features.size() % ((SL_DATA_SIZE_MUL * 3) / 100) == 0) {
                System.out.println(((double) features.size() / (SL_DATA_SIZE_MUL * 3)) * 100 + "%");
            }
        }
        System.out.println("Loaded " + features.size() + " games.");
        System.out.println(w + ", " + b + ", " + t);
        return BoardConverter.convert(features, labels);
    }
}

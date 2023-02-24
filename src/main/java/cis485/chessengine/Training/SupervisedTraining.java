package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.game.Game;
import com.github.bhlangonijr.chesslib.game.GameResult;
import com.github.bhlangonijr.chesslib.move.Move;
import com.github.bhlangonijr.chesslib.move.MoveList;
import com.github.bhlangonijr.chesslib.pgn.GameLoader;
import com.github.bhlangonijr.chesslib.pgn.PgnIterator;
import org.apache.commons.compress.compressors.zstandard.ZstdCompressorOutputStream;
import org.apache.commons.compress.compressors.zstandard.ZstdUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class SupervisedTraining {
    //https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319
    private static final Random RANDOM = new Random();

    private static final int SL_EPOCHS = 50;
    private static final int SL_DATA_SIZE_MUL = 100_000;
    private static final int SL_WHITE_WINS = SL_DATA_SIZE_MUL;
    private static final int SL_BLACK_WINS = SL_DATA_SIZE_MUL;
    private static final int SL_TIES = SL_DATA_SIZE_MUL;
    private static final int SL_MINI_BATCH_SIZE = 64;

    public static void main(String[] args) {
        System.out.println("Supervised training:");
        System.out.println("\tLoading positions...");
        long start = System.nanoTime();
        DataSet data = loadData();
        long end = System.nanoTime();
        double seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished after " + seconds + " seconds");

        // split into training and test data
        data.shuffle();
        SplitTestAndTrain testAndTrain = data.splitTestAndTrain(0.66);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTrain();

        // normalize
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);

        // set up model
        MultiLayerNetwork model = ModelBuilder.Supervised.build();

        // train
        System.out.println("\tBeginning training...");
        start = System.nanoTime();
        model.setInputMiniBatchSize(SL_MINI_BATCH_SIZE);
        for (int i = 0; i < SL_EPOCHS; i++) {
            model.fit(trainingData);
        }
        end = System.nanoTime();
        seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished training after " + seconds + " seconds");

        Evaluation eval = new Evaluation(3);
        eval.eval(testData.getLabels(), model.output(testData.getFeatures()));
        System.out.println(eval.stats());

        try {
            model.save(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SLmodel.mdl"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static DataSet loadData() {
        List<DataSet> data = new ArrayList<>();

        PgnIterator pgnIterator;
        try {
            pgnIterator = new PgnIterator("C:\\Users\\drewm\\Desktop\\EngineTrainingData\\lichess_db_standard_rated_2023-01.pgn");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Iterator<Game> iterator = pgnIterator.iterator();


        while (iterator.hasNext()) {
            Game game = iterator.next();
            int whiteElo = game.getWhitePlayer().getElo();
            int blackElo = game.getBlackPlayer().getElo();
            int moveCount = game.getHalfMoves().size();
            if (whiteElo < 2000 || blackElo < 2000 || moveCount < 1) {
                continue;
            }
            GameResult gameResult = game.getResult();
            float[][] result = new float[1][3];
            switch (gameResult.toString()) {
                case "WHITE_WON":
                    result = new float[][]{{1, 0, 0}};
                    break;
                case "BLACK_WON":
                    result = new float[][]{{0, 1, 0}};
                    break;
                case "TIE":
                    result = new float[][]{{0, 0, 1}};
                    break;
            }
            MoveList moveList = game.getHalfMoves();
            String position = moveList.getFen(RANDOM.nextInt(moveList.size()));
            Board board = new Board();
            board.loadFromFen(position);
            DataSet dataSet = new DataSet(BoardConverter.convert(board, true), Nd4j.create(result));
            data.add(dataSet);
        }
        System.out.println("Loaded " + data.size() + " games.");
        return DataSet.merge(data);
    }

    private static DataSet generateData() {
        List<DataSet> data = new ArrayList<>();
        int w = 0;
        int b = 0;
        int t = 0;
        while (w + b + t < SL_WHITE_WINS + SL_BLACK_WINS + SL_TIES) {
            List<String> positions = new ArrayList<>();
            Board board = new Board();
            while (!board.isMated() && !board.isDraw()) {
                List<Move> legalMoves = board.legalMoves();
                Move move = legalMoves.get(RANDOM.nextInt(legalMoves.size()));
                board.doMove(move);
                positions.add(board.getFen());
            }
            float[][] result = new float[1][3];
            if (board.isDraw()) {
                if (t >= SL_TIES) {
                    continue;
                }
                t++;
                result[0][2] = 1;
            }
            else if (board.isMated()) {
                if (board.getSideToMove() == Side.BLACK) {
                    if (w >= SL_WHITE_WINS) {
                        continue;
                    }
                    w++;
                    result[0][1] = 1;
                }
                else {
                    if (b >= SL_BLACK_WINS) {
                        continue;
                    }
                    b++;
                    result[0][0] = 1;
                }
            }
            Board randomBoard = new Board();
            randomBoard.loadFromFen(positions.get(RANDOM.nextInt(positions.size())));
            DataSet dataSet = new DataSet(BoardConverter.convert(randomBoard, true), Nd4j.create(result));
            data.add(dataSet);
        }
        System.out.println("\tGenerated data set with " + w + " white wins, " + b + " black wins, and " + t + " ties.");
        return DataSet.merge(data);
    }
}

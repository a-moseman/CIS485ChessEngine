package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.game.Game;
import com.github.bhlangonijr.chesslib.game.GameResult;
import com.github.bhlangonijr.chesslib.move.Move;
import com.github.bhlangonijr.chesslib.move.MoveList;
import com.github.bhlangonijr.chesslib.pgn.PgnIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class SupervisedTraining {
    //https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319
    private static final Random RANDOM = new Random();

    private static final int SL_EPOCHS = 1000;
    private static final int SL_DATA_SIZE_MUL = 300_000;
    private static final int SL_WHITE_WINS = SL_DATA_SIZE_MUL;
    private static final int SL_BLACK_WINS = SL_DATA_SIZE_MUL;
    private static final int SL_TIES = SL_DATA_SIZE_MUL;
    private static final int SL_MINI_BATCH_SIZE = 32;
    private static final int SL_MIN_ELO = 1000;

    public static void main(String[] args) {
        System.out.println("Supervised training:");
        System.out.println("\tLoading positions...");
        long start = System.nanoTime();
        DataSet data = loadData();
        data.save(new File("C:\\Users\\drewm\\Desktop\\EngineTrainingData\\training_data-" + System.currentTimeMillis() + ".dat"));
        //DataSet data = new DataSet();
        //data.load(new File("C:\\Users\\drewm\\Desktop\\EngineTrainingData\\training_data-(300_000+elo1000).dat"));
        long end = System.nanoTime();
        double seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished after " + seconds + " seconds");

        System.gc();

        // split into training and test data
        data.shuffle();
        SplitTestAndTrain testAndTrain = data.splitTestAndTrain(0.10);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTrain();

        // normalize
        //DataNormalization normalizer = new NormalizerStandardize();
        //normalizer.fit(trainingData);
        //normalizer.transform(trainingData);
        //normalizer.transform(testData);

        // set up model
        MultiLayerNetwork model = ModelBuilder.Supervised.build();

        // train
        System.out.println("\tBeginning training...");
        start = System.nanoTime();
        model.setInputMiniBatchSize(SL_MINI_BATCH_SIZE);
        double bestValAcc = 0;
        for (int i = 0; i < SL_EPOCHS; i++) {
            model.fit(trainingData);
            System.out.println("Epoch " + (i + 1));
            Evaluation eval = new Evaluation(3);
            eval.eval(testData.getLabels(), model.output(testData.getFeatures()));
            double valAcc = eval.accuracy();
            System.out.println("\tVal ACC: " + valAcc);
            if (valAcc > bestValAcc) {
                bestValAcc = valAcc;
                save(model);
                System.out.println("\tSaved model.");
            }
            System.gc();
        }
        end = System.nanoTime();
        seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished training after " + seconds + " seconds");
    }

    private static void save(MultiLayerNetwork model) {
        try {
            model.save(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SLmodel-best.mdl"));
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

        int w = 0;
        int b = 0;
        int t = 0;

        while (iterator.hasNext() && !(w >= SL_WHITE_WINS && b >= SL_BLACK_WINS && t >= SL_TIES)) {
            Game game = iterator.next();
            int whiteElo = game.getWhitePlayer().getElo();
            int blackElo = game.getBlackPlayer().getElo();
            int moveCount = game.getHalfMoves().size();
            if (whiteElo < SL_MIN_ELO || blackElo < SL_MIN_ELO || moveCount < 1) {
                continue;
            }
            GameResult gameResult = game.getResult();
            float[][] result;
            if (gameResult.toString().equals("WHITE_WON")) {
                if (w >= SL_WHITE_WINS) {
                    continue;
                }
                result = new float[][]{{1, 0, 0}};
                w++;
            }
            else if (gameResult.toString().equals("BLACK_WON")) {
                if (b >= SL_BLACK_WINS) {
                    continue;
                }
                result = new float[][]{{0, 1, 0}};
                b++;
            }
            else {
                if (t >= SL_TIES) {
                    continue;
                }
                result = new float[][]{{0, 0, 1}};
                t++;
            }
            MoveList moveList = game.getHalfMoves();
            String position = moveList.getFen(RANDOM.nextInt(moveList.size()));
            Board board = new Board();
            board.loadFromFen(position);
            DataSet dataSet = new DataSet(BoardConverter.convert(board, true), Nd4j.create(result));
            data.add(dataSet);
            if (data.size() % 10_000 == 0) {
                System.out.println((float) data.size() / (SL_DATA_SIZE_MUL * 3) * 100 + "%");
            }
        }
        System.out.println("Loaded " + data.size() + " games.");
        System.out.println(w + ", " + b + ", " + t);
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

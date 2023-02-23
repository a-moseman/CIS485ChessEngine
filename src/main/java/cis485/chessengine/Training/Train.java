package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Train {
    private static final Random RANDOM = new Random();

    private static final int EPOCHS = 50;
    private static final int WHITE_WINS = 1_000;
    private static final int BLACK_WINS = 1_000;
    private static final int TIES = 1_000;


    public static void main(String[] args) {
        System.out.println("Supervised training:");
        System.out.println("\tGenerating positions...");
        long start = System.nanoTime();
        DataSet data = generateData(WHITE_WINS, BLACK_WINS, TIES);
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
        MultiLayerNetwork model = ModelBuilder.build();
        model.setListeners(new ScoreIterationListener(1));

        // train
        System.out.println("\tBeginning training...");
        start = System.nanoTime();
        for (int i = 0; i < EPOCHS; i++) {
            model.fit(trainingData);
        }
        end = System.nanoTime();
        seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished training after " + seconds + " seconds");

        Evaluation eval = new Evaluation(3);
        eval.eval(testData.getLabels(), model.output(testData.getFeatures()));
        System.out.println(eval.stats());
    }

    private static DataSet generateData(int white, int black, int ties) {
        List<DataSet> data = new ArrayList<>();
        int w, b, t = 0;
        while (w < white && b < black && t < ties) {
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
                if (t >= ties) {
                    continue;
                }
                t++;
                result[0][2] = 1;
            }
            else if (board.isMated()) {
                if (board.getSideToMove() == Side.BLACK) {
                    if (w >= white) {
                        continue;
                    }
                    w++;
                    result[0][1] = 1;
                }
                else {
                    if (b >= black) {
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
        return DataSet.merge(data);
    }
    /*
    private static final int EPOCHS = 100;
    private static final float SECONDS_PER_MOVE = 5;

    public static void main(String[] args) {
        MultiLayerNetwork model = ModelBuilder.build();
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            long start = System.nanoTime();
            System.out.println("Start epoch " + epoch + ".");
            runGame(model);
            long end = System.nanoTime();
            double minutes = (double) (end - start) / 1_000_000_000 / 60;
            System.out.println("Epoch " + epoch + " finished in " + minutes + " minutes.");
        }
        System.out.println("Finished training.");
        try {
            System.out.println("Writing model to file.");
            ModelSerializer.writeModel(model, new File("C:\\Users\\drewm\\Desktop\\EngineModels\\model.mdl"), true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void runGame(MultiLayerNetwork model) {
        System.out.println("\tBegin self-play.");
        Engine engine = new Engine(model);
        engine.setSecondsPerMove(SECONDS_PER_MOVE);
        engine.setTraining(true);
        Board board = new Board();
        while (!board.isMated() && !board.isDraw()) {
            engine.setSide(board.getSideToMove());
            board.doMove(engine.run(board.getFen()));
            System.out.println(board);
            System.out.println(engine.getVisits());
        }
    }

     */
}

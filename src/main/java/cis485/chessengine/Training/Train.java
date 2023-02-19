package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Train {
    private static final Random RANDOM = new Random();
    private static List<INDArray> boardStates = new ArrayList<>();
    private static List<INDArray> results = new ArrayList<>();

    private static final int EPOCHS = 100;
    private static final float SECONDS_PER_MOVE = .1f;

    public static void main(String[] args) {
        // use rollout and train during back propogation?
        MultiLayerNetwork model = ModelBuilder.build();
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.println("Start epoch " + epoch + ".");
            runGame(model);
        }
        try {
            ModelSerializer.writeModel(model, new File("C:\\Users\\drewm\\Desktop\\EngineModels\\model.mdl"), true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void runGame(MultiLayerNetwork model) {
        System.out.println("\tBegin self-play.");
        Engine engine = new Engine(model);
        engine.setSecondsPerMove(SECONDS_PER_MOVE);
        Board board = new Board();
        while (!board.isMated() && !board.isDraw()) {
            engine.setSide(board.getSideToMove());
            board.doMove(engine.run(board.getFen()));
            boardStates.add(BoardConverter.convert(board, false));
        }
        System.out.println("\tBuild training data.");
        float[] result = new float[3];
        if (board.isDraw()) {
            result[2] = 1;
        }
        else if (board.isMated()) {
            if (board.getSideToMove() == Side.WHITE) {
                result[1] = 1; // black win
            }
            else {
                result[0] = 1; // white win
            }
        }
        for (int i = 0; i < boardStates.size(); i++) {
            results.add(Nd4j.create(result));
        }

        while (boardStates.size() > 100) {
            int rand = RANDOM.nextInt(boardStates.size());
            boardStates.remove(rand);
            results.remove(rand);
        }

        // build training data
        INDArray features = Nd4j.create(boardStates.size(), 8, 8, 8);
        INDArray labels = Nd4j.create(boardStates.size(), 3);
        DataSet dataSet = new DataSet(features, labels);
        dataSet.shuffle();
        // train
        System.out.println("\tBegin training.");
        model.fit(dataSet);
        INDArray output = model.output(dataSet.getFeatures());
        Evaluation eval = new Evaluation(3);
        eval.eval(dataSet.getLabels(), output);
        System.out.println(eval.stats());
    }
}

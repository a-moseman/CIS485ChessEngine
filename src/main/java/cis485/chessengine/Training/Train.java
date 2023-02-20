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
    private static final int EPOCHS = 100;
    private static final float SECONDS_PER_MOVE = 1;

    public static void main(String[] args) {
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
        engine.setTraining(true);
        Board board = new Board();
        while (!board.isMated() && !board.isDraw()) {
            engine.setSide(board.getSideToMove());
            board.doMove(engine.run(board.getFen()));
        }
    }
}

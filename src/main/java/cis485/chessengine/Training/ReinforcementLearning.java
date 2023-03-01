package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReinforcementLearning {
    private static final Random RANDOM = new Random();
    private static final int GAMES = 100;
    private static final int SECONDS_PER_MOVE = 1;

    public static void main(String[] args) {
        int wins = 0;
        int losses = 0;
        int draws = 0;

        MultiLayerNetwork oldModel = null;
        MultiLayerNetwork newModel = null;
        try {
            oldModel = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V1.mdl"), false);
            newModel = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V1.mdl"), true);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Engine alphaEngine = new Engine(newModel);
        Engine betaEngine = new Engine(oldModel);
        alphaEngine.setSecondsPerMove(SECONDS_PER_MOVE); // the training engine
        betaEngine.setSecondsPerMove(SECONDS_PER_MOVE); // the opponent engine
        for (int i = 0; i < GAMES; i++) {
            System.out.println("Starting self-play " + (i + 1) + "...");
            // run game
            alphaEngine.setSide(i % 2 == 0 ? Side.WHITE : Side.BLACK);
            betaEngine.setSide(i % 2 == 0 ? Side.BLACK : Side.WHITE);
            Board board = new Board();
            List<String> gamePositions = new ArrayList<>();
            while (!board.isMated() && !board.isDraw()) {
                if (board.getSideToMove() == alphaEngine.getSide()) {
                    board.doMove(alphaEngine.run(board.getFen()));
                }
                else {
                    board.doMove(betaEngine.run(board.getFen()));
                }
                gamePositions.add(board.getFen());
            }
            // train model
            double winRate = (double) wins / (wins + losses + draws);
            System.out.println("Current Win Rate: " + winRate);
            System.out.println("Beginning training...");
            float[][] result;
            if (board.isDraw()) {
                result = new float[][]{{0, 0, 1}};
                draws++;
            }
            else {
                if (board.getSideToMove() == Side.BLACK) {
                    result = new float[][]{{1, 0, 0}};
                }
                else {
                    result = new float[][]{{0, 1, 0}};
                }
                if (alphaEngine.getSide() == board.getSideToMove()) {
                    losses++;
                }
                else {
                    wins++;
                }
            }
            String position = gamePositions.get(RANDOM.nextInt(gamePositions.size()));
            Board tempBoard = new Board();
            tempBoard.loadFromFen(position);
            DataSet dataSet = new DataSet(BoardConverter.convert(tempBoard, true), Nd4j.create(result));
            newModel.fit(dataSet);
            save(newModel);
        }

        System.out.println("Finished.");
    }

    private static void save(MultiLayerNetwork model) {
        try {
            model.save(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\rl.mdl"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

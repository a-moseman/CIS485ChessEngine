package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    private static final int EPOCHS = 50;
    private static final float SECONDS_PER_MOVE = 1f;

    public static void main(String[] args) {
        MultiLayerNetwork model = ModelBuilder.build();
        Engine alpha = new Engine(model);
        Engine beta = new Engine(model);
        TrainingStats trainingStats = new TrainingStats();
        long trainingStartTime = System.currentTimeMillis();
        for (int e = 0; e < EPOCHS; e++) {
            System.out.println("Starting epoch " + e + "...");
            long startTime = System.currentTimeMillis();
            MatchStats matchStats = runMatch(SECONDS_PER_MOVE, alpha, beta);
            long endTime = System.currentTimeMillis();
            trainingStats.addMatch(matchStats, endTime - startTime);
            double minutes = (double) (endTime - startTime) / 1000 / 60;
            System.out.println("Completed epoch " + e + " after " + minutes + " minutes");
        }
        long trainingEndTime = System.currentTimeMillis();
        trainingStats.setTotalTrainingTime(trainingEndTime - trainingStartTime);
        System.out.println("Finished training");
        trainingStats.printResults();
        trainingStats.printStatistics();
        // save to file
        try {
            ModelSerializer.writeModel(model, "C:\\Users\\drewm\\Desktop\\EngineModels\\model.mdl", true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static MatchStats runMatch(float secondsPerMove, Engine alpha, Engine beta) {
        GameStats gameOne = runGame(secondsPerMove, Side.WHITE, alpha, beta);
        GameStats gameTwo = runGame(secondsPerMove, Side.BLACK, alpha, beta);
        return new MatchStats(gameOne, gameTwo);
    }

    private static GameStats runGame(float secondsPerMove, Side alphaSide, Engine alpha, Engine beta) {
        GameStats gameStats = new GameStats(alphaSide == Side.WHITE, secondsPerMove);
        Board board = new Board();
        alpha.setSecondsPerMove(secondsPerMove);
        beta.setSecondsPerMove(secondsPerMove);
        alpha.setSide(alphaSide);
        beta.setSide(alphaSide == Side.WHITE ? Side.BLACK : Side.WHITE);
        long startTime = System.currentTimeMillis();
        List<INDArray> x = new ArrayList<>(); // inputs for training
        while (!board.isDraw() && !board.isMated()) {
            if (board.getSideToMove() == alphaSide) {
                Move move = alpha.run(board.getFen());
                board.doMove(move);
                gameStats.updateFromMove(alpha.getVisits(), 0, move.toString());
            }
            else {
                Move move = beta.run(board.getFen());
                board.doMove(move);
                gameStats.updateFromMove(0, beta.getVisits(), move.toString());
            }
            x.add(BoardConverter.convert(board));
        }
        long endTime = System.currentTimeMillis();
        int result = 0;
        if (board.isMated()) {
            // 1 for white win, -1 for black win
            result = board.getSideToMove() == Side.WHITE ? -1 : 1;
        }
        gameStats.setResult(result, endTime - startTime);

        // train models
        List<float[][]> y = new ArrayList<>(); // labels for training
        for (int i = 0; i < x.size(); i++) {
            if (i % 2 == 0) { // white
                if (result == 1) {
                    y.add(new float[][]{{1, 0}});
                }
                else {
                    y.add(new float[][]{{0, 1}});
                }
            }
            else { // black
                if (result == -1) {
                    y.add(new float[][]{{1, 0}});
                }
                else {
                    y.add(new float[][]{{0, 1}});
                }
            }
        }

        System.out.println("Self-play complete, beginning training...");
        for (int i = 0; i < x.size(); i++) {
            alpha.getModel().fit(x.get(i), Nd4j.create(y.get(i)));
        }

        return gameStats;
    }
}
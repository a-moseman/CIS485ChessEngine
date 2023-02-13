package cis485.chessengine.Training;

import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;

public class Main {
    private static final int EPOCHS = 10;
    private static final int SECONDS_PER_MOVE = 1;
    public static void main(String[] args) {
        Engine alpha = new Engine();
        Engine beta = new Engine();
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
    }

    private static MatchStats runMatch(int secondsPerMove, Engine alpha, Engine beta) {
        GameStats gameOne = runGame(secondsPerMove, Side.WHITE, alpha, beta);
        GameStats gameTwo = runGame(secondsPerMove, Side.BLACK, alpha, beta);
        return new MatchStats(gameOne, gameTwo);
    }

    private static GameStats runGame(int secondsPerMove, Side alphaSide, Engine alpha, Engine beta) {
        GameStats gameStats = new GameStats(alphaSide == Side.WHITE, secondsPerMove);
        Board board = new Board();
        alpha.setSecondsPerMove(secondsPerMove);
        beta.setSecondsPerMove(secondsPerMove);
        long startTime = System.currentTimeMillis();
        while (!board.isDraw() && !board.isMated()) {
            if (board.getSideToMove() == alphaSide) {
                board.doMove(alpha.run(board.getFen()));
                gameStats.updateFromMove(alpha.getVisits(), 0);
            }
            else {
                board.doMove(beta.run(board.getFen()));
                gameStats.updateFromMove(0, beta.getVisits());
            }
        }
        long endTime = System.currentTimeMillis();
        int result = 0;
        if (board.isMated()) {
            // 1 for white win, -1 for black win
            result = board.getSideToMove() == Side.WHITE ? -1 : 1;
        }
        gameStats.setResult(result, endTime - startTime);
        return gameStats;
    }
}
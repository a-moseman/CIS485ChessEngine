package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;

public class Training {
    private static final int EPOCHS = 1;
    private static final int SECONDS_PER_MOVE = 1;
    public static void main(String[] args) {
        Engine alpha = new Engine();
        Engine beta = new Engine();
        int[] results = new int[EPOCHS * 2];
        for (int e = 0; e < EPOCHS; e++) {
            System.out.println("Starting epoch " + e + "...");
            long startTime = System.currentTimeMillis();
            int[] result = runMatch(SECONDS_PER_MOVE, alpha, beta);
            results[e * 2] = result[0];
            results[e * 2 + 1] = result[1];
            long endTime = System.currentTimeMillis();
            double minutes = (double) (endTime - startTime) / 1000 / 60;
            System.out.println("Completed epoch " + e + " after " + minutes + " minutes");
        }
        System.out.println("Finished training");

        int whiteWins = 0;
        int blackWins = 0;
        int ties = 0;
        for (int i = 0; i < results.length; i++) {
            switch (results[i]) {
                case 0:
                    ties++;
                case 1:
                    whiteWins++;
                case -1:
                    blackWins++;
            }
        }
        System.out.println("Training Results:");
        System.out.println("\tWhite Wins: " + whiteWins);
        System.out.println("\tBlack Wins: " + blackWins);
        System.out.println("\tTies: " + ties);
    }

    private static int[] runMatch(int secondsPerMove, Engine alpha, Engine beta) {
        int r1 = runGame(secondsPerMove, alpha, beta);
        int r2 = runGame(secondsPerMove, beta, alpha);
        return new int[]{r1, r2};
    }

    private static int runGame(int secondsPerMove, Engine white, Engine black) {
        Board board = new Board();
        white.setSecondsPerMove(secondsPerMove);
        black.setSecondsPerMove(secondsPerMove);
        while (!board.isDraw() && !board.isMated()) {
            if (board.getSideToMove() == Side.WHITE) {
                board.doMove(white.run(board.getFen()));
            }
            else {
                board.doMove(black.run(board.getFen()));
            }
        }
        if (board.isMated()) {
            // 1 for white win, -1 for black win
            return board.getSideToMove() == Side.WHITE ? -1 : 1;
        }
        return 0; // draw
    }
}
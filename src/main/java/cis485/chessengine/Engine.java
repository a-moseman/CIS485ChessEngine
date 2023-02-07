package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;

public class Engine {
    private int secondsPerMove = 10;

    public void evaluate(String position) {
        Board board = new Board();
        board.loadFromFen(position);
    }

    public int getSecondsPerMove() {
        return secondsPerMove;
    }

    public void setSecondsPerMove(int secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }
}

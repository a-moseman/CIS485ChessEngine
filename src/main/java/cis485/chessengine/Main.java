package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;

public class Main {
    public static void main(String[] args) {
        Engine a = new Engine();
        Engine b = new Engine();
        Board board = new Board();
        while (!board.isDraw() && !board.isMated()) {
            System.out.println(board);
            if (board.getSideToMove() == Side.WHITE) {
                board.doMove(a.run(board.getFen()));
            }
            else {
                board.doMove(b.run(board.getFen()));
            }
        }
    }
}
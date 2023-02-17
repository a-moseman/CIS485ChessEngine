package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;

public class Testing {
    public static void main(String[] args) {
        Board board = new Board();
        Engine white = new Engine(ModelBuilder.build());
        Engine black = new Engine(ModelBuilder.build());
        while (!board.isMated() && !board.isDraw()) {
            System.out.println(board);
            if (board.getSideToMove() == Side.WHITE) {
                board.doMove(white.run(board.getFen()));
            }
            else {
                board.doMove(black.run(board.getFen()));
            }
        }
    }
}

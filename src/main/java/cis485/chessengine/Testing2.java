package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.move.Move;

import java.util.List;
import java.util.Random;

public class Testing2 {
    public static void main(String[] args) {
        Random random = new Random();
        Board board = null;
        boolean running = true;
        while (running) {
            board = new Board();
            while (!board.isDraw() && ! board.isMated()) {
                List<Move> moves = board.legalMoves();
                Move move = moves.get(random.nextInt(moves.size()));
                board.doMove(move);
            }
            if (board.isMated()) {
                running = false;
            }
        }
        System.out.println(board);
        System.out.println(board.getSideToMove());
    }
}

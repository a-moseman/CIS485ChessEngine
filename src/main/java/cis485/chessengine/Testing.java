package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;

public class Testing {
    public static void main(String[] args) {
        Board board = new Board();
        board.loadFromFen("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3");
        long a = System.nanoTime();
        Board b = board.clone();
        System.out.println(System.nanoTime() - a);

        long c = System.nanoTime();
        Board d = new Board();
        d.loadFromFen(board.getFen());
        System.out.println(System.nanoTime() - c);
    }
}

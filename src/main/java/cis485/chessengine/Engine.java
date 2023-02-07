package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

import java.util.List;
import java.util.Random;

public class Engine {
    private Random random;
    private int secondsPerMove = 10;
    private Board board;
    private Side side;

    public Engine() {
        this.board = new Board();
        this.random = new Random();
    }

    //https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/

    public void run(String position) {
        board.loadFromFen(position);
        side = board.getSideToMove();
        // todo
    }

    private int rollout(Board board) {
        if (board.isMated()) {
            return board.getSideToMove() == side ? -1 : 1;
        }
        if (board.isDraw()) {
            return 0;
        }

        List<Move> moves = board.legalMoves();
        Move move = moves.get(random.nextInt(moves.size() - 1));
        board.doMove(move);
        int score = rollout(board);
        board.undoMove();
        return score;
    }

    private int evaluate(Board board) {
        return 0; // todo: apply ml
    }

    public int getSecondsPerMove() {
        return secondsPerMove;
    }

    public void setSecondsPerMove(int secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }
}

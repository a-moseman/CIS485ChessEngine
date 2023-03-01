package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class Testing {
    public static void main(String[] args) {
        MultiLayerNetwork model = null;
        try {
            model = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V1.mdl"), false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Engine engine = new Engine(model);
        Board board = new Board();
        while (!board.isMated() && !board.isDraw()) {
            System.out.println(board);
            if (board.getSideToMove() == Side.WHITE) {
                engine.setSide(board.getSideToMove());
                engine.setSecondsPerMove(10);
                board.doMove(engine.run(board.getFen()));
            }
            else {
                List<Move> legalMoves = board.legalMoves();
                Collections.shuffle(legalMoves);
                Move move = legalMoves.get(0);
                board.doMove(move);
            }
        }
    }
}

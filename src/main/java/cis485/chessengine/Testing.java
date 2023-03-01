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
        engine.setSecondsPerMove(10);
        Board board = new Board();
        while (!board.isMated() && !board.isDraw()) {
            System.out.println(board);
            engine.setSide(board.getSideToMove());
            Move move = engine.run(board.getFen());
            engine.printEvaluations();
            board.doMove(move);
        }
    }
}

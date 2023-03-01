package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.File;
import java.io.IOException;

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
            engine.setSide(board.getSideToMove());
            engine.setSecondsPerMove(10);
            board.doMove(engine.run(board.getFen()));
        }
    }
}

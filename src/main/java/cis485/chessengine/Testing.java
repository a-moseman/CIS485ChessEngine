package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.File;
import java.io.IOException;

public class Testing {
    public static void main(String[] args) {
        Board board = new Board();
        MultiLayerNetwork model = null;
        try {
            model = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\model.mdl"), true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Engine white = new Engine(model);
        Engine black = new Engine(ModelBuilder.build());
        white.setSide(Side.WHITE);
        white.setSecondsPerMove(0.5f);
        black.setSide(Side.BLACK);
        black.setSecondsPerMove(0.5f);
        while (!board.isMated() && !board.isDraw()) {
            System.out.println(board);
            if (board.getSideToMove() == Side.WHITE) {
                board.doMove(white.run(board.getFen()));
                System.out.println("Visited Nodes: " + white.getVisits());
            }
            else {
                board.doMove(black.run(board.getFen()));
                System.out.println("Visited Nodes: " + black.getVisits());
            }
        }
    }
}

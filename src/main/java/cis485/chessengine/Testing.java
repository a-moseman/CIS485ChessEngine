package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;

public class Testing {
    /*
    public static void main(String[] args) {
        float SECONDS = 1;

        Board board = new Board();
        MultiLayerNetwork model = ModelBuilder.build();
        LayerHelper h = model.getLayer(0).getHelper();
        System.out.println(h==null?null:h.getClass().getName());
        //        try {
        //            model = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\model.mdl"), true);
        //        } catch (IOException e) {
        //            throw new RuntimeException(e);
        //        }
        Engine white = new Engine(model);
        Engine black = new Engine(ModelBuilder.build());
        white.setSide(Side.WHITE);
        white.setSecondsPerMove(SECONDS);
        black.setSide(Side.BLACK);
        black.setSecondsPerMove(SECONDS);
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

     */
}

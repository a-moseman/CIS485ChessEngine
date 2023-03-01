package cis485.chessengine;

import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class Testing {
    public static void main(String[] args) {
        MultiLayerNetwork model = null;
        try {
            model = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V1.mdl"), false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Engine engine = new Engine(model);
        engine.setSecondsPerMove(1);
        engine.setSide(Side.WHITE);
        Board board = new Board();
        Scanner scanner = new Scanner(System.in);
        while (!board.isMated() && !board.isDraw()) {
            System.out.println(board);
            if (board.getSideToMove() == Side.WHITE) {
                Move move = engine.run(board.getFen());
                engine.printEvaluations();
                board.doMove(move);
            }
            else {
                List<Move> legalMoves = board.legalMoves();
                for (Move move : legalMoves) {
                    System.out.print(move + ", ");
                }
                System.out.println();
                board.doMove(scanner.nextLine());
            }
        }
    }
}

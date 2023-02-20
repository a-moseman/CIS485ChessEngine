package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.*;

public class UCI {
    private static final String INIT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    private static Engine engine;
    private static String fen = INIT_FEN;

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        MultiLayerNetwork model = ModelBuilder.build();
        engine = new Engine(model);
        engine.setTraining(false);
        boolean running = true;
        while (running) {
            // Get input
            String command = input.nextLine();

            // Quit the program
            if ("quit".equals(command)) running = false;

            else if ("isready".equals(command)) {
                isready();
            } else if (command.startsWith("position")) {
                 position(command);
            } else if ("go".equals(command)) {
                go();
            } else if ("print".equals(command)) {
                print();
            }
        }
    }
    // "used to synchronize the engine with the GUI"
    public static void isready() {
        System.out.println("readyok");
    }
    // "set up the position described in fenstring and play the moves on the internal chess board"
    public static void position(String input) {
        input = input.substring(9).concat(" ");
        if (input.contains("fen")) {
            input = input.substring(4);
            fen = input;
        }
        /* won't implement for prototype
        else if (input.contains("startpos ")) {
            input = input.substring(9);
            // todo give board input
        }
         */
        if (input.contains("moves")) {
            input = input.substring(input.indexOf("moves")+6);
            String[] moves = input.split(" ");
            Board board = new Board();
            for (String move : moves) {
                board.doMove(move);
            }
            fen = board.getFen();
        }
    }
    // "start calculating on the current position set up with the "position" command"
    public static void go() {
        Move bestMove;
        bestMove = engine.run(fen);
        System.out.println("bestmove " + bestMove);
    }
    // Show board
    public static void print() {
        Board board = new Board();
        board.loadFromFen(fen);
        System.out.println(board);
    }
}

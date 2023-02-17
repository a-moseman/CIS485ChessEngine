package cis485.chessengine.Engine;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.*;

public class UCI {
    public static void main(String[] args) {
        MultiLayerNetwork model = ModelBuilder.build();
        Engine engine = new Engine(model);
        Scanner input = new Scanner(System.in);
        // todo - stuff
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
            // todo give board input
        } else if (input.contains("startpos ")) {
            input = input.substring(9);
            // todo give board input
        }
        if (input.contains("moves")) {
            input = input.substring(input.indexOf("moves")+6);
            // todo make moves
        }
    }
    // "start calculating on the current position set up with the "position" command"
    public static void go() {
        // todo
    }
    // Show board
    public static void print() {
        // todo
    }
}

package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.shade.guava.util.concurrent.MoreExecutors;

import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class UCI {
    private static final String INIT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    private static Engine engine;
    private static String fen = INIT_FEN;

    public static void main(String[] args) {
        MultiLayerNetwork model = null;
        //String modelVersion = System.getProperty("user.dir") + "\\src\\main\\java\\cis485\\chessengine\\Engine\\Model\\SL_MODEL_V1.mdl";
        try {
            String name = "res/RL_MODEL_V2.mdl";
            InputStream inputStream = ClassLoader.getSystemClassLoader().getResourceAsStream(name);
            if (inputStream == null) {
                throw new Exception("Improper path to neural network.");
            }
            byte[] bytes = inputStream.readAllBytes();
            inputStream.close();
            String tempFileName = "model_temp_file-" + System.currentTimeMillis() + ".mdl";
            OutputStream outputStream = new FileOutputStream(tempFileName);
            outputStream.write(bytes);
            outputStream.close();
            File file = new File(tempFileName);
            model = MultiLayerNetwork.load(file, false);
            file.delete();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        engine = new Engine(model, true);

        // Thread pool management
        final int MAX_THREADS = 5;
        final int THREAD_HANG_DURATION = 10; //in seconds
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(MAX_THREADS);
        ExecutorService executorService = MoreExecutors.getExitingExecutorService(executor, THREAD_HANG_DURATION, TimeUnit.SECONDS);

        // UCI calls
        Scanner input = new Scanner(System.in);
        boolean running = true;
        while (running) {
            // Get input
            String raw = input.nextLine();
            String[] command = raw.split(" ");

            // Quit the program
            if ("quit".equals(command[0])) {
                running = false;
                System.out.println("Terminating...");
            } else if ("isready".equals(command[0])) {
                //executorService.submit(() -> {
                //    isready();
                //});
                isready();
            } else if (command[0].equals("position")) {
                position(raw);
            } else if ("go".equals(command[0])) {
                //executorService.submit(() -> {
                //    go(command);
                //});
                go(command);
            } else if ("print".equals(command[0])) {
                print();
            }
            /*
            else if ("test".equals(command[0])) {
                System.out.println("running");
                executorService.submit(() -> {
                    while (true) {
                        //test for thread termination
                    }
                });
            }*/
        }

    }

    // Provides synchronization for the engine
    public static void isready() {
        System.out.println("readyok");
    }
    // "set up the position described in fenstring and play the moves on the internal chess board"
    public static void position(String input) {
        input = input.substring(9).concat(" ");
        if (input.contains("fen")) {
            input = input.substring(4);
            fen = input;
            isready();
        }
        /* won't implement for prototype
        else if (input.contains("startpos ")) {
            input = input.substring(9);
            //board input
        } */
        if (input.contains("moves")) {
            input = input.substring(input.indexOf("moves")+6);
            String[] moves = input.split(" ");
            Board board = new Board();
            for (String move : moves) {
                board.doMove(move);
            }
            fen = board.getFen();
            isready();
        }
    }
    // "start calculating on the current position set up with the "position" command"
    public static void go(String[] command) {
        Move bestMove;
        for (int i = 1; i < command.length; i++) {
            if (command[i].equals("movetime")) {
                int seconds = Integer.parseInt(command[i + 1]);
                engine.setSecondsPerMove(seconds);
            }
        }
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

package cis485.chessengine.Engine;

import cis485.chessengine.Engine.Search.MCTS;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class Engine {
    private final MultiLayerNetwork MODEL;
    private float secondsPerMove;
    private Side side;
    private MCTS mcts;

    public Engine(MultiLayerNetwork model) {
        this.MODEL = model;
        this.mcts = new MCTS(MODEL);
        this.secondsPerMove = 10; // default
    }

    /**
     * Run the engine on the position.
     * @param position The position in FEN.
     * @return Move
     */
    public Move run(String position) {
        mcts.initialize(side, position);
        long start = System.nanoTime();
        while (System.nanoTime() - start < 1_000_000_000L * secondsPerMove) {
            mcts.step();
        }
        return mcts.getBest();
    }

    public int getVisits() {
        if (mcts == null) {
            return 0;
        }
        return mcts.getVisits();
    }

    public void setSecondsPerMove(float secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }

    public void setSide(Side side) {
        this.side = side;
    }

    public Side getSide() {
        return side;
    }

    public void printEvaluations() {
        mcts.printEvaluations();
    }
}

package cis485.chessengine.Engine;

import cis485.chessengine.Engine.Search.MCTS;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class Engine {
    private float secondsPerMove = 10;
    private Side side;
    private MultiLayerNetwork model;
    private MCTS mcts;

    public Engine(MultiLayerNetwork model) {
        this.mcts = new MCTS(model);
        this.secondsPerMove = 10; // default
        this.model = model;
    }


    private boolean training;

    public void setTraining(boolean training) {
        this.training = training;
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
        mcts.printEvaluations();
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
}

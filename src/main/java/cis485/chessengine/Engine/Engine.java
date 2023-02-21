package cis485.chessengine.Engine;

import cis485.chessengine.Engine.Search.MCTS;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

public class Engine {
    private float secondsPerMove = 10;
    private Side side;
    //private MultiLayerNetwork model;
    private MCTS mcts;

    /*
    public Engine(MultiLayerNetwork model) {
        this.secondsPerMove = 10; // default
        this.model = model;
    }
    */

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
        //mcts = new MCTS(model, side, position);
        mcts = new MCTS(side, position);
        mcts.setTraining(training);
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

    public float getSecondsPerMove() {
        return secondsPerMove;
    }

    public void setSecondsPerMove(float secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }

    /*
    public MultiLayerNetwork getModel() {
        return model;
    }
     */

    public Side getSide() {
        return side;
    }

    public void setSide(Side side) {
        this.side = side;
    }
}

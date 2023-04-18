package cis485.chessengine.Engine.Search;

import cis485.chessengine.Engine.BoardConverter;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

//https://ai.stackexchange.com/questions/16238/how-is-the-rollout-from-the-mcts-implemented-in-both-of-the-alphago-zero-and-the
public class MCTS {
    private final MultiLayerNetwork MODEL;
    private Side side;
    private Node root;
    private int visits;
    private boolean valueBased;

    public MCTS(MultiLayerNetwork model, boolean valueBased) {
        this.MODEL = model;
        this.valueBased = valueBased;
    }

    public void initialize(Side side, String position) {
        this.side = side;
        Board board = new Board();
        board.loadFromFen(position);
        root = new Node(null, board);
        root.visited = true;
        root.totalVisits = 1;
        visits = 0;
    }


    public void step() {
        visits++;
        Node leaf = traverse(root);
        double reward = predict(leaf.position);
        backpropagate(leaf, reward);
    }

    private double predict(Board position) {
        if (position.isDraw()) {
            return -0.5;
        }
        else if (position.isMated()){
            if (position.getSideToMove() == Side.BLACK) {
                return -1;
            }
            else {
                return 1;
            }
        }
        float[][][][] board = new float[][][][]{BoardConverter.oneHotEncode(position)};
        INDArray input = Nd4j.create(board);
        INDArray output = MODEL.output(input, false);
        if (valueBased) { // for RL model
            return output.toFloatVector()[0];
        }
        else { // for SL model
            float[] out = output.toFloatVector();
            return out[0] - out[1] - out[2];
        }
    }

    public void printEvaluations() {
        System.out.println("Visits: " + visits);
        for (Node node : root.children) {
            System.out.printf("%s: %.2f\n", node.move, node.reward / node.totalVisits);
        }
    }

    private Node traverse(Node node) {
        if (isTerminal(node)) {
            return node;
        }
        while (isFullyExpanded(node)) {
            if (isTerminal(node)) {
                return node;
            }
            node = bestUCT(node); // can only be a visited node
        }
        return pickUnvisited(node);
    }

    private void backpropagate(Node node, double reward) {
        //visits++;
        node.visited = true;
        node.totalVisits++;
        node.reward += reward;
        if (root.equals(node)) {
            return;
        }
        backpropagate(node.parent, -reward);
    }

    private boolean isFullyExpanded(Node node) {
        if (node.children.size() < node.position.legalMoves().size()) {
            return false;
        }
        return true;
    }

    private Node bestUCT(Node node) {
        int best = 0;
        for (int i = 1; i < node.children.size(); i++) {
            if (uctOfChild(node.children.get(best)) < uctOfChild(node.children.get(i))) {
                best = i;
            }
        }
        return node.children.get(best);
    }

    private double uctOfChild(Node child) {
        double c = Math.sqrt(2);
        double exploitationComponent = child.reward / child.totalVisits;
        double explorationComponent = Math.sqrt(Math.log(child.parent.totalVisits) / child.totalVisits);
        return exploitationComponent + c * explorationComponent ;
    }

    private Node pickUnvisited(Node node) {
        List<Move> legalMoves = node.position.legalMoves();
        if (legalMoves.size() == 0) { // node is terminal
            return node;
        }
        // add unvisited child to tree
        Board childBoard = new Board();
        childBoard.loadFromFen(node.position.getFen());
        Move move = legalMoves.get(node.children.size());
        childBoard.doMove(move);
        Node child = new Node(move, childBoard);
        node.children.add(child);
        child.parent = node;
        return child;
    }

    public Move getBest() {
        int best = 0;
        for (int i = 1; i < root.children.size(); i++) {
            if (root.children.get(best).reward < root.children.get(i).reward) {
                best = i;
            }
        }
        return root.children.get(best).move;
    }

    public Node getBestNode() {
        int best = 0;
        for (int i = 1; i < root.children.size(); i++) {
            if (root.children.get(best).reward < root.children.get(i).reward) {
                best = i;
            }
        }
        return root.children.get(best);
    }

    public int getVisits() {
        return visits;
    }

    public boolean isTerminal(Node node) {
        return node.position.isDraw() || node.position.isMated();
    }
}

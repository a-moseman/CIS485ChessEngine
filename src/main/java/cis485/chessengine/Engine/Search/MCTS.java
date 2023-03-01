package cis485.chessengine.Engine.Search;

import cis485.chessengine.Engine.BoardConverter;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.*;

//https://ai.stackexchange.com/questions/16238/how-is-the-rollout-from-the-mcts-implemented-in-both-of-the-alphago-zero-and-the
public class MCTS {
    private final Random RANDOM = new Random();
    private final MultiLayerNetwork MODEL;
    private Side side;
    private Node root;
    private int visits;

    public MCTS(MultiLayerNetwork model) {
        this.MODEL = model;
    }

    private void add(Node node) {
        List<Move> legalMoves = node.position.legalMoves();
        node.children = new Node[legalMoves.size()];
        for (int i = 0; i < node.children.length; i++) {
            Move move = legalMoves.get(i);
            Board newBoard = new Board();
            newBoard.loadFromFen(node.position.getFen()); // faster than Board.clone()
            newBoard.doMove(move);
            Node child = new Node(move, newBoard, node);
            node.children[i] = child;
        }
    }

    public void initialize(Side side, String position) {
        this.side = side;
        Board board = new Board();
        board.loadFromFen(position);
        root = new Node(null, board, null);
        add(root);
        visits = 0;
    }


    public void step() {
        Node leaf = traverse(root);
        Prediction prediction = predict(leaf.position);
        backpropagate(leaf, prediction);
    }

    class Prediction {
        int result;
        double confidence;
    }

    private Prediction predict(Board position) {
        Prediction prediction = new Prediction();
        if (position.isDraw()) {
            prediction.result = 2;
            prediction.confidence = 100;
        }
        else if (position.isMated()){
            if (position.getSideToMove() == Side.BLACK) {
                prediction.result = 0;
            }
            else {
                prediction.result = 1;
            }
            prediction.confidence = 100;
        }

        float[] out = MODEL.output(BoardConverter.convert(position, true)).toFloatVector();
        int pred;
        if (out[0] > out[1] && out[0] > out[2]) { // white wins
            pred = 0;
        }
        else if (out[1] > out[0] && out[1] > out[2]) { // black wins
            pred = 1;
        }
        else { // draw
            pred = 2;
        }
        prediction.result = pred;
        prediction.confidence = out[pred];
        return prediction;
    }

    public void printEvaluations() {
        System.out.println("Visits: " + visits);
        for (Node node : root.children) {
            System.out.printf("%s: %.2f\n", node.move, (double) node.getTotalSimReward(side) / node.totalVisits);
            //System.out.printf("%s: %.2f / %d = %.2f | (%.4f)\n", node.move, node.getTotalSimReward(side), node.totalVisits, ((double) node.getTotalSimReward(side) / node.totalVisits), uctOfChild(node));
        }
    }

    private Node traverse(Node node) {
        while (isFullyExpanded(node)) {
            node = bestUCT(node);
            add(node);
        }
        if (node.children.length == 0) {
            return node;
        }
        return pickUnvisited(node.children);
    }

    private void backpropagate(Node node, Prediction prediction) {
        visits++;
        node.visited = true;
        if (root.equals(node)) {
            return;
        }
        node.totalVisits++;
        switch (prediction.result) {
            case 0:
                node.totalSimWhiteWins += prediction.confidence;
                break;
            case 1:
                node.totalSimBlackWins += prediction.confidence;
                break;
            case 2:
                node.totalSimTies += prediction.confidence;
                break;
        }
        backpropagate(node.parent, prediction);
    }

    private boolean isFullyExpanded(Node node) {
        if (node.children.length == 0) {
            return false;
        }
        for (Node child : node.children) {
            if (!child.visited) {
                return false;
            }
        }
        return true;
    }

    private Node bestUCT(Node node) {
        int best = 0;
        for (int i = 1; i < node.children.length; i++) {
            if (uctOfChild(node.children[best]) < uctOfChild(node.children[i])) {
                best = i;
            }
        }
        return node.children[best];
    }

    private double uctOfChild(Node child) {
        // todo: double check math
        double c = Math.sqrt(2);
        double exploitationComponent = (double) child.getTotalSimReward(side) / child.totalVisits;
        double explorationComponent = Math.sqrt((Math.log(1 + child.parent.totalVisits) + 1) / child.totalVisits);
        return exploitationComponent + c * explorationComponent;
    }

    private Node pickUnvisited(Node[] children) {
        for (Node child : children) {
            if (!child.visited) {
                return child;
            }
        }
        return null;
    }

    public Move getBest() {
        int best = 0;
        for (int i = 1; i < root.children.length; i++) {
            if (root.children[best].getTotalSimReward(side) < root.children[i].getTotalSimReward(side)) {
            //if (root.children[best].totalVisits < root.children[i].totalVisits) {
                best = i;
            }
        }
        return root.children[best].move;
    }

    public int getVisits() {
        return visits;
    }
}

package cis485.chessengine.Engine.Search;

import cis485.chessengine.Engine.BoardConverter;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;
import java.util.Random;

//https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/

//https://int8.io/monte-carlo-tree-search-beginners-guide/

public class MCTS {
    private static final Random RANDOM = new Random();
    private MultiLayerNetwork model;
    private Side side;
    private Node root;
    private int visits;

    public MCTS(MultiLayerNetwork model, Side side, String position) {
        this.model = model;
        this.side = side;
        Board board = new Board();
        board.loadFromFen(position);
        root = new Node(null, board, null);
        visit(root);
        visits = 0;
    }

    private void visit(Node node) {
        visits++; // DEBUG

        node.visited = true;
        List<Move> legalMoves = node.position.legalMoves();
        node.children = new Node[legalMoves.size()];
        int i;
        for (i = 0; i < node.children.length; i++) {
            Move move = legalMoves.get(i);
            Board newBoard = new Board();
            newBoard.loadFromFen(node.position.getFen()); // faster than Board.clone()
            newBoard.doMove(move);
            Node child = new Node(move, newBoard, node);
            node.children[i] = child;
        }
    }

    public void step() {
        Node leaf = traverse(root);
        int result = rollout(leaf);
        //int result = evaluate(leaf);
        backpropagate(leaf, result);
    }

    public Move getBest() {
        return bestChild(root).move;
    }

    private Node traverse(Node node) {
        while (isFullyExpanded(node)) {
            if (node.children.length == 0) { // sometimes runs out of children
                return node;
            }
            node = bestUCT(node);
        }
        return pickUnvisited(node.children);
    }

    private int rollout(Node node) {
        while (isNonTerminal(node)) {
            visit(node);
            node = rollOutPolicy(node);
        }
        return result(node);
    }

    /**
     * Alternative to rollout.
     */
    private int evaluate(Node node) {
        visit(node);
        INDArray x = BoardConverter.convert(node.position);
        int[] y = model.predict(x); // biggest bottle-neck
        return y[0] == 1 ? 1 : -1;
    }

    private void backpropagate(Node node, int result) {
        if (root.equals(node)) {
            return;
        }
        node.totalVisits++;
        node.totalSimReward += result;
        backpropagate(node.parent, result);
    }

    //https://ai.stackexchange.com/questions/16238/how-is-the-rollout-from-the-mcts-implemented-in-both-of-the-alphago-zero-and-the
    private Node rollOutPolicy(Node node) {
        // todo: speed up (78064000 ns atm)
        // get predicted outcomes for each position
        // note: these are game results, not if this engine will win (i.e., they predict which side will win, not if it will win)
        int best = 0;
        INDArray x;
        INDArray y;
        int i;
        float[][] predictions = new float[node.children.length][2];
        int s = side == Side.WHITE ? 0 : 1;
        int notS = side == Side.WHITE ? 1 : 0;
        for (i = 0; i < node.children.length; i++) {
            x = BoardConverter.convert(node.children[i].position);
            y = model.output(x);
            predictions[i][0] = y.getFloat(0);
            predictions[i][1] = y.getFloat(1);
            if (predictions[i][s] < predictions[i][notS]) { // ignore where other side is predicted better
                continue;
            }
            if (i > 0) {
                if (predictions[best][s] < predictions[i][s]) {
                    best = i;
                }
            }
        }
        return node.children[best];
    }

    private boolean isNonTerminal(Node node) {
        return !node.position.isDraw() && !node.position.isMated();
    }

    private boolean isFullyExpanded(Node node) {
        for (Node child : node.children) {
            if (!child.visited) {
                return false;
            }
        }
        return true;
    }

    private Node bestUCT(Node node) {
        double bestUct = uctOfChild(node.children[0]);
        int best = 0;
        for (int i = 0; i < node.children.length; i++) {
            double uct = uctOfChild(node.children[i]);
            if (bestUct < uct) {
                bestUct = uct;
                best = i;
            }
        }
        return node.children[best];
    }

    private double uctOfChild(Node child) {
        double c = Math.sqrt(2);
        double exploitationComponent = (double) child.totalSimReward / child.totalVisits;
        double explorationComponent = Math.sqrt(Math.log(child.parent.totalVisits) / child.totalVisits);
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

    private Node bestChild(Node node) {
        int best = 0;
        int i;
        for (i = 1; i < node.children.length; i++) {
            if (node.children[best].totalVisits < node.children[i].totalVisits) {
                best = i;
            }
        }
        return node.children[best];
    }

    private int result(Node node) {
        if (node.position.isMated()) {
            if (node.position.getSideToMove() != side) {
                return 1;
            }
        }
        return -1; // if loss or draw
    }

    public int getVisits() {
        return visits;
    }
}

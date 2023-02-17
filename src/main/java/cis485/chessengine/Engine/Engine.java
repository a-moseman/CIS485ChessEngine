package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Random;

public class Engine {
    private Random random;
    private int secondsPerMove;
    private Board board;
    private Side side;
    private MultiLayerNetwork model;

    public Engine(MultiLayerNetwork model) {
        this.secondsPerMove = 10; // default
        this.board = new Board();
        this.random = new Random();
        this.model = model;
    }

    //https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/

    class Node {
        Node parent;
        Node[] children;
        Board position;
        Move move;
        int totalSimReward;
        int totalVisits;
        boolean visited;

        public Node(Move move, Board position, Node parent) {
            this.move = move;
            this.position = position;
            this.parent = parent;
        }
    }

    Node root;
    int visits;
    public int getVisits() {
        return visits;
    }

    //https://int8.io/monte-carlo-tree-search-beginners-guide/

    public Move run(String position) {
        visits = 0; // DEBUG

        board.loadFromFen(position);
        side = board.getSideToMove();
        root = new Node(null, board, null);
        visit(root);
        Move best = mcts(root);
        return best;
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

    private Move mcts(Node root) {
        long startTime = System.nanoTime();
        while (System.nanoTime() - startTime < 1_000_000_000L * secondsPerMove) {
            Node leaf = traverse(root);
            int simResult = rollout(leaf);
            backpropagate(leaf, simResult);
        }
        return bestChild(root).move;
    }

    private Node traverse(Node node) {
        while (isFullyExpanded(node)) {
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

    private void backpropagate(Node node, int result) {
        if (root.equals(node)) {
            return;
        }
        node.totalVisits++;
        node.totalSimReward += result;
        backpropagate(node.parent, result);
    }

    private Node rollOutPolicy(Node node) {
        // get predicted outcomes for each position
        // note: these are game results, not if this engine will win (i.e., they predict which side will win, not if it will win)
        int best = 0;
        INDArray x;
        INDArray y;
        int i;
        float[][] predictions = new float[node.children.length][2];
        int s = side == Side.WHITE ? 0 : 1;
        for (i = 0; i < node.children.length; i++) {
            x = Nd4j.create(BoardConverter.convert(node.children[i].position));
            y = model.output(x);
            predictions[i][0] = y.getFloat(0);
            predictions[i][1] = y.getFloat(1);
            if (predictions[i][s] < predictions[i][Math.abs(s - 1)]) { // ignore where other side is predicted better
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

    public int getSecondsPerMove() {
        return secondsPerMove;
    }

    public void setSecondsPerMove(int secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }
}

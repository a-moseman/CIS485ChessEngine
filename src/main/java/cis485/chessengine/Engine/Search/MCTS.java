package cis485.chessengine.Engine.Search;

import cis485.chessengine.Engine.BoardConverter;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.*;

public class MCTS {
    private static final Random RANDOM = new Random();
    private MultiLayerNetwork model;
    private Side side;
    private Node root;
    private int visits;

    public MCTS(MultiLayerNetwork model) {
        this.model = model;
    }

    public void initialize(Side side, String position) {
        // todo
        this.side = side;
        Board board = new Board();
        board.loadFromFen(position);
        if (root == null) {
            root = new Node(null, board, null);
            visit(root);
            visits = 0;
        }
        else {
            // check if new position is child of old tree
            for (int i = 0; i < root.children.length; i++) {
                if (root.children[i].position.getZobristKey() == board.getZobristKey()) {
                    root = root.children[i];
                    if (!root.visited) {
                        visit(root);
                    }
                    return;
                }
            }
            // not of old tree, so make new one
            root = new Node(null, board, null);
            visit(root);
            visits = 0;
        }
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
        int result = rollout(leaf.position);
        backpropagate(leaf, result);
    }

    public Move getBest() {
        return bestChild(root).move;
    }

    public void printEvaluations() {
        for (Node node : root.children) {
            System.out.println(node.getTotalSimReward(side));
        }
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

    private int rollout(Board position) {
        while (isNonTerminal(position)) {
            position = rollOutPolicy(position.getFen());
        }
        return result(position);
    }

    private void backpropagate(Node node, int result) {
        if (root.equals(node)) {
            return;
        }
        node.totalVisits++;
        switch (result) {
            case 0:
                node.totalSimTies++;
                break;
            case 1:
                node.totalSimWhiteWins++;
                break;
            case -1:
                node.totalSimBlackWins++;
                break;
        }
        backpropagate(node.parent, result);
    }

    class Eval implements Comparable<Eval> {
        int index;
        float prediction;

        @Override
        public int compareTo(Eval o) {
            if (prediction == o.prediction) {
                return 0;
            }
            else if (prediction < o.prediction) {
                return 1;
            }
            else {
                return -1;
            }
        }
    }

    //https://ai.stackexchange.com/questions/16238/how-is-the-rollout-from-the-mcts-implemented-in-both-of-the-alphago-zero-and-the
    private Board rollOutPolicy(String fen) {
        Board position = new Board();
        position.loadFromFen(fen);
        List<Move> legalMoves = position.legalMoves();
        List<Eval> evals = new ArrayList<>();
        for (int i = 0; i < legalMoves.size(); i++) {
            position.doMove(legalMoves.get(i));
            Eval eval = new Eval();
            eval.prediction = getValue(model.output(BoardConverter.convert(position, true)).toFloatVector());
            eval.index = i;
            evals.add(eval);
            position.undoMove();
        }
        Collections.sort(evals);
        double r = RANDOM.nextDouble();
        int chosen = 0;
        for (int i = 0; i < evals.size(); i++) {
            double t = 1d / (i + 2);
            if (r > t) {
                chosen = i;
                break;
            }
        }
        position.doMove(legalMoves.get(chosen));
        return position;
    }

    /**
     * Solely for MCTS.rollOutPolicy.
     * @param prediction The output of the model.
     * @return float
     */
    private float getValue(float[] prediction) {
        if (side == Side.WHITE) {
            return prediction[0] - prediction[1] - prediction[2]; // white - black - draw
        }
        else {
            return prediction[1] - prediction[0] - prediction[2]; // black - white - draw
        }
    }

    private boolean isNonTerminal(Board position) {
        return !position.isDraw() && !position.isMated();
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
        double exploitationComponent = (double) child.getTotalSimReward(side) / child.totalVisits;
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
            if (node.children[best].getTotalSimReward(side) < node.children[i].getTotalSimReward(side)) {
                best = i;
            }
        }
        return node.children[best];
    }

    private int result(Board position) {
        if (position.isMated()) {
            if (position.getSideToMove() != Side.WHITE) {
                return 1;
            }
            else {
                return -1;
            }
        }
        return 0;
    }

    public int getVisits() {
        return visits;
    }
}

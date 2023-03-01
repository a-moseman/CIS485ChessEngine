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
        int result = predict(leaf.position);
        backpropagate(leaf, result);
    }

    private int predict(Board position) {
        return MODEL.predict(BoardConverter.convert(position, true))[0];
    }

    public void printEvaluations() {
        System.out.println("Visits: " + visits);
        for (Node node : root.children) {
            System.out.println(node.move + ": " + ((double) node.getTotalSimReward(side) / node.totalVisits) );
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

    private void backpropagate(Node node, int result) {
        visits++;
        node.visited = true;
        if (root.equals(node)) {
            return;
        }
        node.totalVisits++;
        switch (result) {
            case 0:
                node.totalSimWhiteWins++;
                break;
            case 1:
                node.totalSimBlackWins++;
                break;
            case 2:
                node.totalSimTies++;
                break;
        }
        backpropagate(node.parent, result);
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
        int best = RANDOM.nextInt(node.children.length);
        for (int i = 0; i < node.children.length; i++) {
            if (uctOfChild(node.children[best]) < uctOfChild(node.children[i])) {
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

    public Move getBest() {
        int best = RANDOM.nextInt(root.children.length);
        int i;
        for (i = 0; i < root.children.length; i++) {
            //if (root.children[best].getTotalSimReward(side) < root.children[i].getTotalSimReward(side)) {
            if (root.children[best].totalVisits < root.children[i].totalVisits) {
                best = i;
            }
        }
        return root.children[best].move;
    }

    public int getVisits() {
        return visits;
    }
}

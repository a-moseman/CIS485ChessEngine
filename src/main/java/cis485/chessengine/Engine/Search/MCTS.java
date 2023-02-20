package cis485.chessengine.Engine.Search;

import cis485.chessengine.Engine.BoardConverter;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

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
    private boolean training;

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
        if (training) {
            INDArray feature = BoardConverter.convert(node.position, false);
            float[] raw = new float[3];
            switch (result) {
                case 0:
                    raw[2] = 1;
                    break;
                case 1:
                    raw[0] = 1;
                    break;
                case -1:
                    raw[1] = 1;
                    break;
            }
            INDArray label = Nd4j.create(raw);
            DataSet dataSet = new DataSet(feature, label);
            model.fit(dataSet);
        }
        backpropagate(node.parent, result);
    }

    //https://ai.stackexchange.com/questions/16238/how-is-the-rollout-from-the-mcts-implemented-in-both-of-the-alphago-zero-and-the
    private Node rollOutPolicy(Node node) {
        int best = 0;
        INDArray[] y = new INDArray[node.children.length];
        for (int i = 0; i < node.children.length; i++) {
            Node child = node.children[i];
            y[i] = model.output(BoardConverter.convert(child.position, true));
            if (getValue(y[i].toFloatVector()) > getValue(y[best].toFloatVector())) {
                best = i;
            }
        }
        //return node.children[RANDOM.nextInt(node.children.length)];
        return node.children[best];
    }

    private float getValue(float[] prediction) {
        if (side == Side.WHITE) {
            return prediction[0] - prediction[1] - prediction[2]; // white - black - draw
        }
        else {
            return prediction[1] - prediction[0] - prediction[2]; // black - white - draw
        }
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

    private int result(Node node) {
        if (node.position.isMated()) {
            if (node.position.getSideToMove() != Side.WHITE) {
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

    public void setTraining(boolean training) {
        this.training = training;
    }
}

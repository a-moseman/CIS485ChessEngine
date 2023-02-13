package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

import java.util.List;
import java.util.Random;

public class Engine {
    private Random random;
    private int secondsPerMove = 10;
    private Board board;
    private Side side;

    public Engine() {
        this.board = new Board();
        this.random = new Random();
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
        visits = 0;

        board.loadFromFen(position);
        side = board.getSideToMove();
        root = new Node(null, board, null);
        visit(root);
        Move best = mcts(root);

        // DEBUG
        /*
        for (int i = 0; i < root.children.length; i++) {
            Move move = root.children[i].move;
            int v = root.children[i].totalVisits;
            int r = root.children[i].totalSimReward;
            System.out.println(move + " (" + r + ", " + v + ")");
        }
        System.out.println("Nodes: " + visits);
         */
        // DEBUG

        return best;
    }

    private void visit(Node node) {
        visits++;

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
            //System.out.println(leaf.move + " - " + simResult); // DEBUG
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
        // DONE
        while (isNonTerminal(node)) {
            visit(node);
            node = rollOutPolicy(node);
        }
        return result(node);
    }

    private void backpropagate(Node node, int result) {
        // DONE
        if (root.equals(node)) {
            return;
        }
        node.totalVisits++;
        node.totalSimReward += result;
        backpropagate(node.parent, result);
    }

    private Node rollOutPolicy(Node node) {
        // TODO: impelement ml stuff?
        return node.children[random.nextInt(node.children.length)]; // pick random
        /*
        int bestScore = Evaluator.materialBalance(node.children.get(0).position, side);
        int best = 0;
        for (int i = 0; i < node.children.size(); i++) {
            int score = Evaluator.materialBalance(node.children.get(i).position, side);
            if (bestScore < score) {
                bestScore = score;
                best = i;
            }
        }
        return node.children.get(best);
         */
    }

    private boolean isNonTerminal(Node node) {
        // DONE
        return !node.position.isDraw() && !node.position.isMated();
    }

    private boolean isFullyExpanded(Node node) {
        // DONE
        for (Node child : node.children) {
            if (!child.visited) {
                return false;
            }
        }
        return true;
    }

    private Node bestUCT(Node node) {
        // DONE
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
        // DONE
        double c = Math.sqrt(2);
        double exploitationComponent = (double) child.totalSimReward / child.totalVisits;
        double explorationComponent = Math.sqrt(Math.log(child.parent.totalVisits) / child.totalVisits);
        return exploitationComponent + c * explorationComponent;
    }

    private Node pickUnvisited(Node[] children) {
        // DONE
        for (Node child : children) {
            if (!child.visited) {
                return child;
            }
        }
        return null;
    }

    private Node bestChild(Node node) {
        // DONE
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
            else {
                return -1;
            }
        }
        return 0; // draw
    }

    public int getSecondsPerMove() {
        return secondsPerMove;
    }

    public void setSecondsPerMove(int secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }
}

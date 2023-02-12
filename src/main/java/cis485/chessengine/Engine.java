package cis485.chessengine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Engine {
    private Random random;
    private int secondsPerMove = 2;
    private Board board;
    private Side side;

    public Engine() {
        this.board = new Board();
        this.random = new Random();
    }

    //https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/

    class Node {
        Node parent;
        List<Node> children;
        Board position;
        Move move;
        int totalSimReward;
        int totalVisits;
        boolean visited;

        public Node(Move move, Board position, Node parent) {
            this.move = move;
            this.position = position;
            this.parent = parent;
            this.children = new ArrayList<>();
        }
    }

    Node root;

    //https://int8.io/monte-carlo-tree-search-beginners-guide/

    public Move run(String position) {
        board.loadFromFen(position);
        side = board.getSideToMove();
        root = new Node(null, board, null);
        visit(root);
        return mcts(root);
    }

    private void visit(Node node) {
        node.visited = true;
        List<Move> legalMoves = node.position.legalMoves();
        for (Move move : legalMoves) {
            Board newBoard = node.position.clone();
            newBoard.doMove(move);
            Node child = new Node(move, newBoard, node);
            node.children.add(child);
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
        return pickRandom(node.children);
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
        return !node.position.isMated() && !node.position.isDraw();
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
        double bestUct = uctOfChild(node.children.get(0), node);
        int best = 0;
        for (int i = 0; i < node.children.size(); i++) {
            double uct = uctOfChild(node.children.get(i), node);
            if (bestUct < uct) {
                bestUct = uct;
                best = i;
            }
        }
        return node.children.get(best);
    }

    private double uctOfChild(Node child, Node parent) {
        // DONE
        double c = 0.5;
        double exploitationComponent = (double) child.totalSimReward / child.totalVisits;
        double explorationComponent = Math.sqrt(Math.log(parent.totalVisits) / child.totalVisits);
        return exploitationComponent + c * explorationComponent;
    }

    private Node pickUnvisited(List<Node> children) {
        // DONE
        for (Node child : children) {
            if (!child.visited) {
                return child;
            }
        }
        return null;
    }

    private Node pickRandom(List<Node> children) {
        // DONE
        int choice = random.nextInt(children.size());
        return children.get(choice);
    }

    private Node bestChild(Node node) {
        // DONE
        int best = 0;
        for (int i = 1; i < node.children.size(); i++) {
            if (node.children.get(best).totalSimReward < node.children.get(i).totalSimReward) {
                best = i;
            }
        }
        return node.children.get(best);
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
        if (node.position.isDraw()) {
            return -1;
        }
        return 0; // should never happen
    }

    public int getSecondsPerMove() {
        return secondsPerMove;
    }

    public void setSecondsPerMove(int secondsPerMove) {
        this.secondsPerMove = secondsPerMove;
    }
}

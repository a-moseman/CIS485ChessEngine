package cis485.chessengine.Engine;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.move.Move;

public class Node {
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

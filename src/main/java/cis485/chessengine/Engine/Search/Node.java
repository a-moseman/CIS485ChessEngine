package cis485.chessengine.Engine.Search;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.move.Move;

public class Node {
    public Node parent;
    public Node[] children;
    public Board position;
    public Move move;
    public int totalSimReward;
    public int totalVisits;
    public boolean visited;

    public Node(Move move, Board position, Node parent) {
        this.move = move;
        this.position = position;
        this.parent = parent;
    }
}

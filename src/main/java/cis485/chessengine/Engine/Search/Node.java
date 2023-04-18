package cis485.chessengine.Engine.Search;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

import java.util.ArrayList;
import java.util.List;

public class Node {
    public Node parent;
    public List<Node> children;
    public Board position;
    public Move move;
    public int totalVisits;
    double reward;
    public boolean visited;

    public Node(Move move, Board position) {
        this.move = move;
        this.position = position;
        this.parent = null;
        this.children = new ArrayList<>();
    }
}

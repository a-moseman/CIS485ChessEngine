package cis485.chessengine.Engine.Search;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

import java.util.ArrayList;
import java.util.List;

public class Node {
    public List<Node> parents;
    public List<Node> children;
    public Board position;
    public Move move;
    public int totalVisits;
    public double totalSimWhiteWins;
    public double totalSimBlackWins;
    public double totalSimTies;
    public boolean visited;

    public Node(Move move, Board position) {
        this.move = move;
        this.position = position;
        this.parents = new ArrayList<>();
        this.children = new ArrayList<>();
    }

    public double getTotalSimReward(Side side) {
        if (side == Side.WHITE) {
            return totalSimWhiteWins - totalSimBlackWins - totalSimTies;
        }
        else {
            return totalSimBlackWins - totalSimWhiteWins - totalSimTies;
        }
    }

    public double getTotalParentVisits() {
        double v = 0;
        for (Node parent : parents) {
            v += parent.totalVisits;
        }
        return v / parents.size();
    }
}

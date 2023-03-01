package cis485.chessengine.Engine.Search;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;

public class Node {
    public Node parent;
    public Node[] children;
    public Board position;
    public Move move;
    public int totalVisits;
    public double totalSimWhiteWins;
    public double totalSimBlackWins;
    public double totalSimTies;
    public boolean visited;

    public Node(Move move, Board position, Node parent) {
        this.move = move;
        this.position = position;
        this.parent = parent;
    }

    public double getTotalSimReward(Side side) {
        if (side == Side.WHITE) {
            return totalSimWhiteWins - totalSimBlackWins - totalSimTies;
        }
        else {
            return totalSimBlackWins - totalSimWhiteWins - totalSimTies;
        }
    }
}

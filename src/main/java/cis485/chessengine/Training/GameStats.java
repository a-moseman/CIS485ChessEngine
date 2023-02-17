package cis485.chessengine.Training;

import java.util.ArrayList;
import java.util.List;

public class GameStats {
    public final boolean IS_ALPHA_WHITE;
    public final float SECONDS_PER_MOVE;
    private int moves;
    private int alphaTotalNodesVisited;
    private int betaTotalNodesVisited;
    private int result;
    private long gameDuration;
    private List<String> moveList;

    public GameStats(boolean isAlphaWhite, float secondsPerMove) {
        this.IS_ALPHA_WHITE = isAlphaWhite;
        this.SECONDS_PER_MOVE = secondsPerMove;
        this.moveList = new ArrayList<>();
    }

    public void updateFromMove(int alphaNodes, int betaNodes, String move) {
        moves++;
        alphaTotalNodesVisited += alphaNodes;
        betaTotalNodesVisited += betaNodes;
        moveList.add(move);
    }

    public void setResult(int result, long gameDuration) {
        this.result = result;
        this.gameDuration = gameDuration;
    }

    public int getMoves() {
        return moves;
    }

    public int getAlphaTotalNodesVisited() {
        return alphaTotalNodesVisited;
    }

    public int getBetaTotalNodesVisited() {
        return betaTotalNodesVisited;
    }

    public int getResult() {
        return result;
    }

    public long getGameDuration() {
        return gameDuration;
    }

    private String getMoveListAsString() {
        StringBuilder sb = new StringBuilder();
        for (String move : moveList) {
            sb.append(move).append(',');
        }
        return sb.toString();
    }
}
package cis485.chessengine.Training;

public class GameStats {
    public final boolean IS_ALPHA_WHITE;
    public final int SECONDS_PER_MOVE;
    private int moves;
    private int alphaTotalNodesVisited;
    private int betaTotalNodesVisited;
    private int result;
    private long gameDuration;

    public GameStats(boolean isAlphaWhite, int secondsPerMove) {
        this.IS_ALPHA_WHITE = isAlphaWhite;
        this.SECONDS_PER_MOVE = secondsPerMove;
    }

    public void updateFromMove(int alphaNodes, int betaNodes) {
        moves++;
        alphaTotalNodesVisited += alphaNodes;
        betaTotalNodesVisited += betaNodes;
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
}
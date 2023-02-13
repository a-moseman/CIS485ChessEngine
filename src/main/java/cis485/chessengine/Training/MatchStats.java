package cis485.chessengine.Training;

public class MatchStats {
    public final GameStats GAME_ONE; // alpha as white
    public final GameStats GAME_TWO; // alpha as black

    public MatchStats(GameStats gameOne, GameStats gameTwo) {
        this.GAME_ONE = gameOne;
        this.GAME_TWO = gameTwo;
    }
}

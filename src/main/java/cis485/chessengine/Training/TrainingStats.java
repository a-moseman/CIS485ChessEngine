package cis485.chessengine.Training;

import java.util.ArrayList;
import java.util.List;

public class TrainingStats {
    private List<MatchStats> matches;
    private List<Long> trueMatchTimes;
    private long totalTrainingTime;

    public TrainingStats() {
        this.matches = new ArrayList<>();
        this.trueMatchTimes = new ArrayList<>();
        this.totalTrainingTime = 0;
    }

    public void addMatch(MatchStats matchStats, long duration) {
        matches.add(matchStats);
        trueMatchTimes.add(duration);
    }

    public void setTotalTrainingTime(long totalTrainingTime) {
        this.totalTrainingTime = totalTrainingTime;
    }

    public void printResults() {
        int alphaWins = 0;
        int betaWins = 0;
        int ties = 0;
        for (int i = 0; i < matches.size(); i++) {
            MatchStats matchStats = matches.get(i);
            GameStats gameOne = matchStats.GAME_ONE;
            GameStats gameTwo = matchStats.GAME_TWO;
            if (gameOne.getResult() == 0) {
                ties++;
            }
            if (gameTwo.getResult() == 0) {
                ties++;
            }
            if (gameOne.getResult() == 1) {
                if (gameOne.IS_ALPHA_WHITE) {
                    alphaWins++;
                }
                else {
                    betaWins++;
                }
            }
            if (gameTwo.getResult() == 1) {
                if (gameOne.IS_ALPHA_WHITE) {
                    alphaWins++;
                }
                else {
                    betaWins++;
                }
            }
            if (gameOne.getResult() == -1) {
                if (gameOne.IS_ALPHA_WHITE) {
                    betaWins++;
                }
                else {
                    alphaWins++;
                }
            }
            if (gameTwo.getResult() == -1) {
                if (gameTwo.IS_ALPHA_WHITE) {
                    betaWins++;
                }
                else {
                    alphaWins++;
                }
            }
        }
        System.out.println("Alpha Wins: " + alphaWins);
        System.out.println("Beta Wins: " + betaWins);
        System.out.println("Ties: " + ties);
    }

    public void printStatistics() {
        double averageAlphaNodesVisited = 0;
        double averageBetaNodesVisited = 0;
        for (int i = 0; i < matches.size(); i++) {
            MatchStats matchStats = matches.get(i);
            GameStats gameOne = matchStats.GAME_ONE;
            GameStats gameTwo = matchStats.GAME_TWO;
            averageAlphaNodesVisited += gameOne.getAlphaTotalNodesVisited();
            averageAlphaNodesVisited += gameTwo.getAlphaTotalNodesVisited();
            averageBetaNodesVisited += gameOne.getBetaTotalNodesVisited();
            averageBetaNodesVisited += gameTwo.getBetaTotalNodesVisited();
        }
        averageAlphaNodesVisited /= matches.size();
        averageBetaNodesVisited /= matches.size();
    }
}

package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.Engine;
import cis485.chessengine.Engine.ModelBuilder;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ReinforcementLearning {
    private static final Random RANDOM = new Random();
    private static final int EPISODES = Integer.MAX_VALUE;
    private static final float SECONDS_PER_MOVE = 1f;
    private static final float DISCOUNT_FACTOR = 0.99f;
    private static final float REWARD_FACTOR = 1f;
    private static final int MAX_GAME_LENGTH = 80;
    private static final int BATCH_SIZE = 1;
    private static final int MINI_BATCH_SIZE = 800;
    private static final int MAX_EXPERIENCE_REPLAY_SIZE = 128_000;
    private static final int C = 20;

    private static final double EPS_START = 1;
    private static final double EPS_END = 0.1;
    private static final double EPS_DECAY = 0.001;

    public static void main(String[] args) {
        double eps = EPS_START;
        int draws = 0;
        int wins = 0;
        int losses = 0;

        //https://www.baeldung.com/cs/reinforcement-learning-neural-network
        //https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/transfer-learning
        //https://stackoverflow.com/questions/47036246/dqn-q-loss-not-converging
        MultiLayerNetwork adversaryModel = null;
        try {
            adversaryModel = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V2.mdl"), false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        MultiLayerNetwork trainingModel = ModelBuilder.Reinforcement.build();
        CollectScoresListener collectScoresListener = new CollectScoresListener(1);
        trainingModel.setListeners(collectScoresListener);

        MultiLayerNetwork targetModel = trainingModel.clone();

        Engine adversaryEngine = new Engine(adversaryModel, false);
        Engine trainingEngine = new Engine(trainingModel, true);
        adversaryEngine.setSecondsPerMove(SECONDS_PER_MOVE);
        trainingEngine.setSecondsPerMove(SECONDS_PER_MOVE);
        List<Integer> indices = new ArrayList<>();
        List<float[][][]> positions = new ArrayList<>();
        List<float[]> results = new ArrayList<>();
        List<Float> depthFactor = new ArrayList<>();
        for (int episode = 0; episode < EPISODES; episode++) {
            if (episode % C == 0) {
                targetModel = trainingModel.clone();
            }

            int episodeWins = 0;
            int episodeLosses = 0;
            int episodeDraws = 0;

            System.out.println("Episode " + (episode + 1) + ": ");
            System.out.println("\tStarting self play with epsilon of " + eps);
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                Board board = new Board();
                Side trainingSide = RANDOM.nextBoolean() ? Side.WHITE : Side.BLACK;
                trainingEngine.setSide(trainingSide);
                adversaryEngine.setSide(trainingSide == Side.WHITE ? Side.BLACK : Side.WHITE);
                int movesMade = 0;
                boolean forcedDraw = false;
                while (!board.isMated() && !board.isDraw()) {
                    if (movesMade >= MAX_GAME_LENGTH) {
                        forcedDraw = true;
                        break;
                    }
                    Move move;

                    if (RANDOM.nextDouble() < eps) {
                        List<Move> legalMoves = board.legalMoves();
                        move = legalMoves.get(RANDOM.nextInt(legalMoves.size()));
                    }
                    else {
                        if (trainingSide == board.getSideToMove()) {
                            move = trainingEngine.run(board.getFen());
                            //trainingEngine.printEvaluations();
                            //System.out.println(trainingEngine.getVisits());
                        }
                        else {
                            move = adversaryEngine.run(board.getFen());
                            //System.out.println(adversaryEngine.getVisits());
                        }
                    }
                    board.doMove(move);
                    positions.add(BoardConverter.oneHotEncode(board));
                    movesMade++;
                }
                float[] result = new float[3];
                if (board.isDraw() || forcedDraw) {
                    result[2] = 1;
                    draws++;
                    episodeDraws++;
                }
                else {
                    if (board.getSideToMove() != trainingSide) {
                        wins++;
                        episodeWins++;
                    }
                    else {
                        losses++;
                        episodeLosses++;
                    }

                    if (board.getSideToMove() == Side.BLACK) { // white win
                        result[0] = 1;
                    }
                    else {
                        result[1] = 1;
                    }
                }
                for (int i = 0; i < movesMade; i++) {
                    results.add(result);
                    depthFactor.add((float) (i + 1) / (movesMade + 1));
                }
            }
            while (positions.size() > MAX_EXPERIENCE_REPLAY_SIZE) {
                int i = RANDOM.nextInt(positions.size());
                positions.remove(i);
                results.remove(i);
                depthFactor.remove(i);
            }

            System.out.println("\tSelf-play finished.");
            double episodeWinrate = (double) episodeWins / (episodeWins + episodeLosses + episodeDraws);
            System.out.println("\tEpisode w/l/d: " + episodeWins + "/" + episodeLosses + "/" + episodeDraws + ". (" + episodeWinrate + ")");
            double winrate = (double) wins / (wins + losses + draws);
            System.out.println("\tOverall w/l/d: " + wins + "/" + losses + "/" + draws + ". (" + winrate + ")");

            // train
            System.out.println("\tExperience Replay Size: " + positions.size());
            for (int i = indices.size(); i < positions.size(); i++) {
                indices.add(i);
            }

            Collections.shuffle(indices);
            List<float[][][]> features = new ArrayList<>();
            List<float[]> labels = new ArrayList<>();
            double avgReward = 0;
            for (int i = 0; i < Math.min(positions.size(), MINI_BATCH_SIZE); i++) {
                int index = indices.get(i);
                features.add(positions.get(index));
                INDArray input = Nd4j.create(new float[][][][]{positions.get(index)});
                float prediction = targetModel.output(input).toFloatVector()[0];
                avgReward += prediction;
                float[] target = new float[]{getReward(results.get(index), prediction) * depthFactor.get(i) * REWARD_FACTOR + prediction * DISCOUNT_FACTOR};
                labels.add(target);
            }
            avgReward = avgReward / Math.min(positions.size(), MINI_BATCH_SIZE);
            System.out.println("\tAverage Reward: " + avgReward);
            DataSet data = BoardConverter.convert(features, labels);
            System.out.println("\tStarting training...");
            trainingModel.setInputMiniBatchSize(Math.min(positions.size(), MINI_BATCH_SIZE));
            trainingModel.fit(data);
            System.out.println("\tTraining finished.");
            double loss = collectScoresListener.getListScore().getDouble(collectScoresListener.getListScore().size() - 1);
            System.out.println("Loss: " + loss);

            eps -= EPS_DECAY;
            if (eps < EPS_END) {
                eps = EPS_END;
            }

            save(trainingModel);

            try {
                FileWriter fw = new FileWriter("C:\\Users\\drewm\\Desktop\\EngineModels\\RL_model_training_graph.csv", true);
                BufferedWriter bw = new BufferedWriter(fw);
                bw.write(wins + "," + losses + "," + draws + "," + eps + "," + avgReward + ", " + loss);
                bw.newLine();
                bw.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.gc();
        }
    }

    private static float getReward(float[] result, float prediction) {
        if (result[0] == 1) {
            return 1;
        }
        else if (result[1] == 1) {
            return -1;
        }
        else {
            if (prediction > 0) {
                return -0.1f;
            }
            else if (prediction < 0) {
                return 0.1f;
            }
            return 0;
        }
    }

    private static void save(MultiLayerNetwork model) {
        try {
            model.save(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\RLmodel-last.mdl"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

package cis485.chessengine.Training;

import cis485.chessengine.Engine.BoardConverter;
import cis485.chessengine.Engine.Engine;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import com.github.bhlangonijr.chesslib.move.Move;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReinforcementLearning {
    private static final Random RANDOM = new Random();
    private static final int EPOCHS = 10;
    private static final int SECONDS_PER_MOVE = 1;

    public static void main(String[] args) {
        int draws = 0;
        int wins = 0;
        int losses = 0;

        //https://www.baeldung.com/cs/reinforcement-learning-neural-network
        MultiLayerNetwork adversaryModel = null;
        MultiLayerNetwork pretrainedModel = null;
        try {
            pretrainedModel = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V2.mdl"), true);
            adversaryModel = MultiLayerNetwork.load(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SL_MODEL_V2.mdl"), false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        FineTuneConfiguration conf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(1e-3))
                .dropOut(0.5)
                .build();

        MultiLayerNetwork trainingModel = new TransferLearning.Builder(pretrainedModel)
                .fineTuneConfiguration(conf)
                //.setFeatureExtractor(pretrainedModel.getLayers().length - 1)
                .removeOutputLayer()
                .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(12)
                        .nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU).build()
                ).build();

        Engine adversaryEngine = new Engine(adversaryModel);
        Engine trainingEngine = new Engine(trainingModel);
        adversaryEngine.setSecondsPerMove(SECONDS_PER_MOVE);
        trainingEngine.setSecondsPerMove(SECONDS_PER_MOVE);
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.println("Epoch " + (epoch + 1) + ": ");
            // play game
            System.out.println("\tStarting self play...");
            List<String> positions = new ArrayList<>();
            Board board = new Board();
            Side trainingSide = RANDOM.nextBoolean() ? Side.WHITE : Side.BLACK;
            trainingEngine.setSide(trainingSide);
            adversaryEngine.setSide(trainingSide == Side.WHITE ? Side.BLACK : Side.WHITE);
            while (!board.isMated() || !board.isDraw()) {
                Move move;
                if (trainingSide == board.getSideToMove()) {
                    move = trainingEngine.run(board.getFen());
                }
                else {
                    move = adversaryEngine.run(board.getFen());
                }
                positions.add(board.getFen());
                board.doMove(move);
            }
            float[] result = new float[3];
            if (board.isDraw()) {
                result[2] = 1;
                draws++;
            }
            else {
                if (board.getSideToMove() != trainingSide) {
                    wins++;
                }
                else {
                    losses++;
                }

                if (board.getSideToMove() == Side.BLACK) { // white win
                    result[0] = 1;
                }
                else {
                    result[1] = 1;
                }
            }
            double winrate = (double) wins / (wins + losses + draws);
            System.out.println("\tSelf play finished. Overall w/l/d: " + wins + "/" + losses + "/" + draws + ". (" + winrate + ")");


            // train
            String position = positions.get(RANDOM.nextInt(positions.size()));
            Board tempBoard = new Board();
            tempBoard.loadFromFen(position);


            float discountFactor = 0.95f;
            INDArray input = Nd4j.create(new float[][][][]{BoardConverter.oneHotEncode(board)});
            float[] prediction = trainingModel.output(input).toFloatVector();
            int action = maxarg(prediction);
            float target = result[action] * 0.08f + prediction[action] * discountFactor;
            float[] target_vector = prediction.clone();
            target_vector[action] = target;
            trainingModel.fit(input, Nd4j.create(new float[][]{target_vector}));
        }
    }

    private static float[] multiply(float[] vector, float mul) {
        float[] c = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            c[i] = vector[i] * mul;
        }
        return c;
    }

    private static float[] sum(float[] a, float[] b) {
        assert a.length == b.length;
        float[] c = new float[a.length];
        for (int i = 0; i < c.length; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    private static int maxarg(float[] vector) {
        int arg = 2;
        for (int i = 1; i >= 0; i--) {
            if (vector[arg] < vector[i]) {
                arg = i;
            }
        }
        return arg;
    }
}

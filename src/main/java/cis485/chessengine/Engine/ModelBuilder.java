package cis485.chessengine.Engine;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelBuilder {
    public static class Supervised {
        private static final int CLASSES = 3;
        private static final int WIDTH = 8;
        private static final int HEIGHT = 8;
        private static final int CHANNELS = 12;
        private static final double LEARNING_RATE = 1e-3;
        //private static final double WEIGHT_DECAY = 1e-5;
        private static final double DROPOUT = 0.5;
        private static final int LAYER_BLOCKS = 8;
        private static final int FILTERS = 96;

        public static MultiLayerNetwork build() {
            Layer[] layers = new Layer[LAYER_BLOCKS * 2 + 2];
            for (int i = 0; i < LAYER_BLOCKS; i++) {
                layers[i * 2] = new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .nOut(FILTERS)
                        .stride(1, 1)
                        .padding(2, 2)
                        .build();
                layers[i * 2 + 1] = new BatchNormalization.Builder().build();
            }
            layers[layers.length - 2] = new DenseLayer.Builder()
                    .nOut(FILTERS)
                    .build();
            layers[layers.length - 1] = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nOut(CLASSES)
                    .activation(Activation.SOFTMAX)
                    .build();
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam(LEARNING_RATE))
                    .weightInit(WeightInit.RELU)
                    .activation(Activation.RELU)
                    //.weightDecay(WEIGHT_DECAY)
                    .dropOut(1 - DROPOUT)
                    .list(layers)
                    .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
                    .build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            return model;
        }
    }

    //https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html
    //https://stackoverflow.com/questions/62405053/simple-reinforcement-learning-example

}

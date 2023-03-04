package cis485.chessengine.Engine;

import org.bytedeco.opencv.opencv_dnn.FlattenLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;

public class ModelBuilder {
    // TODO: explore not using pooling layers like AlphaZero and ConvChess
    public static class Supervised {
        private static final int CLASSES = 3;
        private static final int WIDTH = 8;
        private static final int HEIGHT = 8;
        private static final int CHANNELS = 12;
        private static final double LEARNING_RATE = 1e-3;
        private static final double WEIGHT_DECAY = 0.00001;
        private static final double DROPOUT = 0.2;

        private static final int LAYER_BLOCKS = 8;
        private static final int FILTERS = 128;



        public static MultiLayerNetwork build() {
            Layer[] layers = new Layer[8 * 2 + 1];
            for (int i = 0; i < LAYER_BLOCKS; i++) {
                layers[i * 2] = new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .nOut(FILTERS)
                        .stride(1, 1)
                        .build();
                layers[i * 2 + 1] = new BatchNormalization.Builder().build();
            }
            layers[layers.length - 1] = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nOut(CLASSES)
                    .activation(Activation.SOFTMAX)
                    .build();


            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam(LEARNING_RATE))
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .weightDecay(WEIGHT_DECAY)
                    //.dropOut(1 - DROPOUT)
                    .list(layers)
                    .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
                    .build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            return model;
        }
    }

    public static class Reinforcement {
        public static MultiLayerNetwork build() {
            // todo
            return null;
        }
    }

}

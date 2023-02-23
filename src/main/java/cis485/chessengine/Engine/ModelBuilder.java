package cis485.chessengine.Engine;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelBuilder {
    public static class Supervised {
        private static int CHANNELS = 8; // white, black, pawn, rook, ..., king
        private static double LEARNING_RATE = 0.01;
        private static double REGULARIZATION = 0.001;
        private static double DROPOUT = 0.2;

        public static MultiLayerNetwork build() {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam(LEARNING_RATE))
                    .weightInit(WeightInit.NORMAL) // todo: explore
                    .activation(Activation.SIGMOID) // todo: explore
                    .l2(REGULARIZATION)
                    .dropOut(1 - DROPOUT)
                    .list()
                    .layer(new ConvolutionLayer.Builder(5, 5)
                            .nIn(CHANNELS)
                            .stride(1, 1)
                            .nOut(256)
                            .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(new ConvolutionLayer.Builder(3, 3)
                            .nIn(CHANNELS)
                            .stride(1, 1)
                            .padding(2, 2)
                            .nOut(256)
                            .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(new ConvolutionLayer.Builder(3, 3)
                            .nIn(CHANNELS)
                            .stride(1, 1)
                            .nOut(256)
                            .padding(1, 1)
                            .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nOut(1024)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nOut(3)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutional(8, 8, CHANNELS))
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

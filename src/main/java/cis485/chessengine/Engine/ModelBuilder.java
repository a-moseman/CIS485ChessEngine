package cis485.chessengine.Engine;

import org.bytedeco.opencv.opencv_dnn.FlattenLayer;
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
import org.nd4j.linalg.api.ops.impl.shape.Flatten2D;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelBuilder {
    public static class Supervised {
        private static final int CLASSES = 3;
        private static final int WIDTH = 8;
        private static final int HEIGHT = 8;
        private static final int CHANNELS = 8; // white, black, pawn, rook, ..., king
        private static final double LEARNING_RATE = 1e-3;
        private static final double WEIGHT_DECAY = 0.00001;
        private static final double DROPOUT = 0.2;

        public static MultiLayerNetwork build() {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Adam(LEARNING_RATE))
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .weightDecay(WEIGHT_DECAY)
                    //.dropOut(1 - DROPOUT)
                    .list()
                    .layer(new ConvolutionLayer.Builder(3, 3)
                            .nIn(CHANNELS)
                            .stride(1, 1)
                            .nOut(64)
                            .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(new ConvolutionLayer.Builder(3, 3)
                            .stride(1, 1)
                            .padding(2, 2)
                            .nOut(64)
                            .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    /*
                    .layer(new ConvolutionLayer.Builder(3, 3)
                            .stride(1, 1)
                            .padding(2, 2)
                            .nOut(16)
                            .build())
                    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())

                     */
                    .layer(new DenseLayer.Builder()
                            .nOut(256)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nOut(256)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nOut(CLASSES)
                            .activation(Activation.SOFTMAX)
                            .build())
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

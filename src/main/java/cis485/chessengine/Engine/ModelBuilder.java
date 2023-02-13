package cis485.chessengine.Engine;

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
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelBuilder {
    private static int SEED = 1234;
    private static int CHANNELS = 9; // is piece, white, black, pawn, rook, ..., king

    public static MultiLayerNetwork build() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(WeightInit.XAVIER) // todo: explore
                .activation(Activation.RELU) // todo: explore
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(8) // todo: explore
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(64)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(1)
                        .activation(Activation.HARDTANH) // todo: test
                        .build())
                .setInputType(InputType.convolutional(8, 8, CHANNELS))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }
}

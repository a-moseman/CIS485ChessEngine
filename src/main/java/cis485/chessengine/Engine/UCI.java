package cis485.chessengine.Engine;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class UCI {
    public static void main(String[] args) {
        MultiLayerNetwork model = ModelBuilder.build();
        Engine engine = new Engine(model);
        // todo
    }
}

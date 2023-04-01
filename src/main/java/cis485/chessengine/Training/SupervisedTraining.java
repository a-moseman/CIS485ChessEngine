package cis485.chessengine.Training;

import cis485.chessengine.Engine.ModelBuilder;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.VertxUIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class SupervisedTraining {
    //https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319

    private static final int EPOCHS = 500;
    private static final int MINI_BATCH_SIZE = 512;

    public static void main(String[] args) {
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(5_000);

        System.out.println("Supervised training:");
        System.out.println("\tLoading positions...");
        long start = System.nanoTime();
        DataSet data = SLDataLoader.generate();
        //data.save(new File("C:\\Users\\drewm\\Desktop\\EngineTrainingData\\training_data-" + System.currentTimeMillis() + ".dat"));
        //DataSet data = new DataSet();
        //data.load(new File("C:\\Users\\drewm\\Desktop\\EngineTrainingData\\training_data-(300_000+elo2000)-v3(CPU).dat"));
        long end = System.nanoTime();
        double seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished after " + seconds + " seconds");
        System.out.println("Dataset Records: " + data.getLabels().size(0));

        // split into training and test data
        data.shuffle();
        SplitTestAndTrain testAndTrain = data.splitTestAndTrain(0.66);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet validationData = testAndTrain.getTrain();

        // set up model
        MultiLayerNetwork model = ModelBuilder.Supervised.build();

        setUpUI(model);
        model.setInputMiniBatchSize(MINI_BATCH_SIZE);
        train(model, trainingData, validationData, EPOCHS, MINI_BATCH_SIZE);
    }

    private static void train(MultiLayerNetwork model, DataSet trainingData, DataSet validationData, int epochs, int minibatches) {
        // train
        System.out.println("\tBeginning training...");
        long start = System.nanoTime();

        model.setInputMiniBatchSize(minibatches);
        DataSetIterator trainingIterator = new ViewIterator(trainingData, minibatches);
        double bestValAcc = 0;
        for (int i = 0; i < epochs; i++) {
            System.gc();
            model.fit(trainingIterator);
            System.out.println("Epoch " + (i + 1) + " finished.");
            double valAcc = validate(model, validationData);
            System.out.println("\tVal ACC: " + valAcc);
            if (valAcc > bestValAcc) {
                bestValAcc = valAcc;
                save(model);
                System.out.println("\tSaved model.");
            }
        }
        long end = System.nanoTime();
        double seconds = (double) (end - start) / 1_000_000_000;
        System.out.println("\tFinished training after " + seconds + " seconds");
    }

    private static void save(MultiLayerNetwork model) {
        try {
            model.save(new File("C:\\Users\\drewm\\Desktop\\EngineModels\\SLmodel-best.mdl"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void setUpUI(MultiLayerNetwork model) {
        // set up the ui: http://localhost:9000/train/overview
        VertxUIServer uiServer = VertxUIServer.getInstance(9000, false, null);
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        try {
            uiServer.start();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        model.setListeners(new StatsListener(statsStorage));
    }

    private static double validate(MultiLayerNetwork model, DataSet data) {
        DataSetIterator iterator = new ViewIterator(data, MINI_BATCH_SIZE);
        double validationAccuracy = 0;
        int i = 0;
        while (iterator.hasNext()) {
            System.gc();
            i++;
            DataSet v = iterator.next();
            Evaluation evaluation = new Evaluation(3);
            evaluation.eval(v.getLabels(), model.output(v.getFeatures()));
            validationAccuracy += evaluation.accuracy();
        }
        System.out.println(validationAccuracy + " / " + i); // DEBUG
        return validationAccuracy / i;
    }
}

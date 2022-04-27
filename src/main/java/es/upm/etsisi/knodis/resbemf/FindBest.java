package es.upm.etsisi.knodis.resbemf;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.DataSet;
import es.upm.etsisi.cf4j.data.RandomSplitDataSet;
import es.upm.etsisi.cf4j.util.optimization.GridSearchCV;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.CummulativeCoverage;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.CummulativeMAE;
import es.upm.etsisi.knodis.resbemf.recommender.*;

public class FindBest {

    private static String DATASET = "ml100k";

    private static long SEED = 4815162342L;

    public static void main (String[] args) throws Exception {

        DataSet dataset = null;

        if (DATASET.equals("ml100k")) {
            dataset = new RandomSplitDataSet("datasets/ml100k.txt", "\t");
        }

        DataModel datamodel = new DataModel(dataset);

        ParamsGrid paramsGrid = null;
        GridSearchCV search  = null;

        // ResBeMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("regularization", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("scores", new double[]{1, 2, 3, 4, 5});
        paramsGrid.addFixedParam("seed", SEED);

        search = new GridSearchCV(datamodel, paramsGrid, ResBeMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 4, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/resbemf-best.csv");


        // BeMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("regularization", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("ratings", new double[]{1, 2, 3, 4, 5});
        paramsGrid.addFixedParam("seed", SEED);

        search = new GridSearchCV(datamodel, paramsGrid, BeMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 4, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/bemf-best.csv");


        // DirMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("regularization", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("ratings", new double[]{1, 2, 3, 4, 5});
        paramsGrid.addFixedParam("seed", SEED);

        search = new GridSearchCV(datamodel, paramsGrid, DirMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 4, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/dirmf-best.csv");


        // PMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("gamma", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addParam("lambda", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("seed", SEED);

        search = new GridSearchCV(datamodel, paramsGrid, PMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 4, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/pmf-best.csv");


        // MLP

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numEpochs", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("seed", SEED);

        search = new GridSearchCV(datamodel, paramsGrid, MLP.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 4, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/mlp-best.csv");
    }
}

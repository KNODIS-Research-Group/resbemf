package es.upm.etsisi.knodis.resbemf;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.DataSet;
import es.upm.etsisi.cf4j.data.RandomSplitDataSet;
import es.upm.etsisi.cf4j.util.optimization.GridSearchCV;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import es.upm.etsisi.cf4j.util.optimization.RandomSearchCV;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.CummulativeCoverage;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.CummulativeMAE;
import es.upm.etsisi.knodis.resbemf.recommender.*;

public class Optimize {

    private static String DATASET = "ml100k";

    private static double RANDOM_SEARCH_COVERAGE = 1.0;

    private static long SEED = 4815162342L;

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;

        if (DATASET.equals("ml100k")) {
            //dataset = new RandomSplitDataSet("datasets/ml100k.txt", "\t");
            datamodel = BenchmarkDataModels.MovieLens100K();
        }


        ParamsGrid paramsGrid = null;
        RandomSearchCV search  = null;

        // ResBeMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.001, 0.002, 0.003, 0.004, 0.005});
        paramsGrid.addParam("regularization", new double[]{0.01, 0.05, 0.10, 0.15, 0.20});
        paramsGrid.addFixedParam("scores", new double[]{1, 2, 3, 4, 5});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, ResBeMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/resbemf.csv");


        // BeMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.01, 0.02, 0.03, 0.04, 0.05});
        paramsGrid.addParam("regularization", new double[]{0.0001, 0.001, 0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("ratings", new double[]{1, 2, 3, 4, 5});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, BeMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/bemf.csv");


        // DirMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.1, 0.2, 0.3, 0.4, 0.5});
        paramsGrid.addParam("regularization", new double[]{0.0001, 0.00025, 0.0005, 0.00075, 0.001});
        paramsGrid.addFixedParam("ratings", new double[]{1, 2, 3, 4, 5});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, DirMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/dirmf.csv");


        // PMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("gamma", new double[]{0.01, 0.02, 0.03, 0.04, 0.05});
        paramsGrid.addParam("lambda", new double[]{0.0001, 0.001, 0.01, 0.1});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, PMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/pmf.csv");


        // MLP

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numEpochs", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, MLP.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/gridsearch/" + DATASET + "/mlp.csv");
    }
}

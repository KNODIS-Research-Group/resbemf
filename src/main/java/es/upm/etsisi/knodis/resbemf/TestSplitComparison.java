package es.upm.etsisi.knodis.resbemf;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.util.plot.LinePlot;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.Coverage;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.MAE;
import es.upm.etsisi.knodis.resbemf.recommender.*;

public class TestSplitComparison {

    private static String DATASET = "ml100k";

    private static double[] scores;

    private static double maxDiff;

    private static double[] RELIABILITIES = {0.00, 0.05, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95};

    private static long SEED = 4815162342L;

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;

        ResBeMF resbemfMae = null;
        ResBeMF resbemfCoverage = null;
        BeMF bemfMae = null;
        BeMF bemfCoverage = null;
        DirMF dirmfMae = null;
        DirMF dirmfCoverage = null;
        PMF pmf = null;
        MLP mlp = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
            scores = new double[]{1, 2, 3, 4, 5};
            maxDiff = 4.0;

            resbemfMae = new ResBeMF(datamodel, 6, 100, 0.002, 0.2, scores, SEED);
            resbemfCoverage = new ResBeMF(datamodel, 4, 50, 0.002, 0.05, scores, SEED);

            bemfMae = new BeMF(datamodel, 2, 75, 0.02, 1.0, scores, SEED);
            bemfCoverage = new BeMF(datamodel, 10, 100, 0.03, 0.0001, scores, SEED);

            dirmfMae = new DirMF(datamodel, 4, 75, 0.2, 0.001, scores, SEED);
            dirmfCoverage = new DirMF(datamodel, 10, 100, 0.1, 0.0001, scores, SEED);

            pmf = new PMF(datamodel, 6, 100, 0.1, 0.01, SEED);

            mlp = new MLP(datamodel, 2, 100, 0.1, SEED);

        }

        LinePlot maePlot = new LinePlot(RELIABILITIES, "reliability", "1-mae");
        LinePlot coveragePlot = new LinePlot(RELIABILITIES, "reliability", "1-coverage");

        // ResBeMF

        resbemfMae.fit();

        maePlot.addSeries("ResBeMF_MAE");
        coveragePlot.addSeries("ResBeMF_MAE");

        for (double rel : RELIABILITIES) {
            double mae = new MAE(resbemfMae, rel).getScore();
            maePlot.setValue("ResBeMF_MAE", rel, 1-mae/maxDiff);

            double coverage = new Coverage(resbemfMae, rel).getScore();
            coveragePlot.setValue("ResBeMF_MAE", rel, coverage);
        }

        resbemfCoverage.fit();

        maePlot.addSeries("ResBeMF_Coverage");
        coveragePlot.addSeries("ResBeMF_Coverage");

        for (double rel : RELIABILITIES) {
            double mae = new MAE(resbemfCoverage, rel).getScore();
            maePlot.setValue("ResBeMF_Coverage", rel, 1-mae/maxDiff);

            double coverage = new Coverage(resbemfCoverage, rel).getScore();
            coveragePlot.setValue("ResBeMF_Coverage", rel, coverage);
        }

        // BeMF

        bemfMae.fit();

        maePlot.addSeries("BeMF_MAE");
        coveragePlot.addSeries("BeMF_MAE");

        for (double rel : RELIABILITIES) {
            double mae = new MAE(bemfMae, rel).getScore();
            maePlot.setValue("BeMF_MAE", rel, 1-mae/maxDiff);

            double coverage = new Coverage(bemfMae, rel).getScore();
            coveragePlot.setValue("BeMF_MAE", rel, coverage);
        }

        bemfCoverage.fit();

        maePlot.addSeries("BeMF_Coverage");
        coveragePlot.addSeries("BeMF_Coverage");

        for (double rel : RELIABILITIES) {
            double mae = new MAE(bemfCoverage, rel).getScore();
            maePlot.setValue("BeMF_Coverage", rel, 1-mae/maxDiff);

            double coverage = new Coverage(bemfCoverage, rel).getScore();
            coveragePlot.setValue("BeMF_Coverage", rel, coverage);
        }

        // DirMF

        dirmfMae.fit();

        maePlot.addSeries("DirMF_MAE");
        coveragePlot.addSeries("DirMF_MAE");

        for (double rel : RELIABILITIES) {
            double mae = new MAE(dirmfMae, rel).getScore();
            maePlot.setValue("DirMF_MAE", rel, 1-mae/maxDiff);

            double coverage = new Coverage(dirmfMae, rel).getScore();
            coveragePlot.setValue("DirMF_MAE", rel, coverage);
        }

        dirmfCoverage.fit();

        maePlot.addSeries("DirMF_Coverage");
        coveragePlot.addSeries("DirMF_Coverage");

        for (double rel : RELIABILITIES) {
            double mae = new MAE(dirmfCoverage, rel).getScore();
            maePlot.setValue("DirMF_Coverage", rel, 1-mae/maxDiff);

            double coverage = new Coverage(dirmfCoverage, rel).getScore();
            coveragePlot.setValue("DirMF_Coverage", rel, coverage);
        }

        // PMF

        pmf.fit();

        maePlot.addSeries("PMF");
        coveragePlot.addSeries("PMF");

        double pmfMae = new MAE(pmf, 0).getScore();
        double pmfCoverage = new Coverage(pmf, 0).getScore();

        for (double rel : RELIABILITIES) {
            maePlot.setValue("PMF", rel, 1-pmfMae/maxDiff);
            coveragePlot.setValue("PMF", rel, pmfCoverage);
        }

        // MLP

        mlp.fit();

        maePlot.addSeries("MLP");
        coveragePlot.addSeries("MLP");

        double mlpMae = new MAE(mlp, 0).getScore();
        double mlpCoverage = new Coverage(mlp, 0).getScore();

        for (double rel : RELIABILITIES) {
            maePlot.setValue("MLP", rel, 1-mlpMae/maxDiff);
            coveragePlot.setValue("MLP", rel, mlpCoverage);
        }

        // Export results

        maePlot.exportData("results/test-split/" + DATASET + "/mae.csv");
        coveragePlot.exportData("results/test-split/" + DATASET + "/coverage.csv");
    }
}

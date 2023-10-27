package es.upm.etsisi.knodis.resbemf;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.util.plot.LinePlot;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.*;
import es.upm.etsisi.knodis.resbemf.recommender.*;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class TestSplitError {

    private static final String DATASET = "anime";

    private static double[] RELIABILITIES = {0.00, 0.05, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95};

    private static int NUMBER_OF_RECOMMENDATIONS = 10; // for the NDCG score

    private static long SEED = 4815162342L;

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;
        double[] scores = null;
        double relevantScore = 0;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
            scores = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
            relevantScore = 4.0;
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
            scores = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
            relevantScore = 4.0;
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
            scores = new double[]{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
            relevantScore = 3.0;
        } else if (DATASET.equals("anime")) {
            datamodel = BenchmarkDataModels.MyAnimeList();
            scores = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            relevantScore = 7.0;
        }

        double maxDiff = scores[scores.length-1] - scores[0];

        LinePlot maePlot = new LinePlot(RELIABILITIES, "reliability", "1-mae");
        LinePlot coveragePlot = new LinePlot(RELIABILITIES, "reliability", "1-coverage");
        LinePlot accuracyPlot = new LinePlot(RELIABILITIES, "reliability", "accuracy");
        LinePlot mapPlot = new LinePlot(RELIABILITIES, "reliability", "map");

        // ResBeMF

        List<String[]> paretoFront = getParetoFront("resbemf");
        String[] columns = paretoFront.get(0);
        paretoFront.remove(0);

        for (int i = 0; i < paretoFront.size(); i++) {
            String[] record = paretoFront.get(i);

            int numIters = Integer.parseInt(record[getColumnIndex(columns, "numIters")]);
            int numFactors = Integer.parseInt(record[getColumnIndex(columns, "numFactors")]);
            double learningRate = Double.parseDouble(record[getColumnIndex(columns, "learningRate")]);
            double regularization = Double.parseDouble(record[getColumnIndex(columns, "regularization")]);

            ResBeMF resbemf = new ResBeMF(datamodel, numFactors, numIters, learningRate, regularization, scores, SEED);
            resbemf.fit();

            String seriesName = "ResBeMF_" + i;

            maePlot.addSeries(seriesName);
            coveragePlot.addSeries(seriesName);
            accuracyPlot.addSeries(seriesName);
            mapPlot.addSeries(seriesName);

            for (double rel : RELIABILITIES) {
                double mae = new MAE(resbemf, rel).getScore();
                maePlot.setValue(seriesName, rel, 1-mae/maxDiff);

                double coverage = new Coverage(resbemf, rel).getScore();
                coveragePlot.setValue(seriesName, rel, coverage);

                double accuracy = new ClassificationAccuracy(resbemf, rel).getScore();
                accuracyPlot.setValue(seriesName, rel, accuracy);

                double map = new MAP(resbemf, NUMBER_OF_RECOMMENDATIONS, relevantScore, rel).getScore();
                mapPlot.setValue(seriesName, rel, map);
            }
        }


        // BeMF

        paretoFront = getParetoFront("bemf");
        columns = paretoFront.get(0);
        paretoFront.remove(0);

        for (int i = 0; i < paretoFront.size(); i++) {
            String[] record = paretoFront.get(i);

            int numIters = Integer.parseInt(record[getColumnIndex(columns, "numIters")]);
            int numFactors = Integer.parseInt(record[getColumnIndex(columns, "numFactors")]);
            double learningRate = Double.parseDouble(record[getColumnIndex(columns, "learningRate")]);
            double regularization = Double.parseDouble(record[getColumnIndex(columns, "regularization")]);

            BeMF bemf = new BeMF(datamodel, numFactors, numIters, learningRate, regularization, scores, SEED);
            bemf.fit();

            String seriesName = "BeMF_" + i;

            maePlot.addSeries(seriesName);
            coveragePlot.addSeries(seriesName);
            accuracyPlot.addSeries(seriesName);
            mapPlot.addSeries(seriesName);

            for (double rel : RELIABILITIES) {
                double mae = new MAE(bemf, rel).getScore();
                maePlot.setValue(seriesName, rel, 1-mae/maxDiff);

                double coverage = new Coverage(bemf, rel).getScore();
                coveragePlot.setValue(seriesName, rel, coverage);

                double accuracy = new ClassificationAccuracy(bemf, rel).getScore();
                accuracyPlot.setValue(seriesName, rel, accuracy);

                double map = new MAP(bemf, NUMBER_OF_RECOMMENDATIONS, relevantScore, rel).getScore();
                mapPlot.setValue(seriesName, rel, map);
            }
        }


        // DirMF

        paretoFront = getParetoFront("dirmf");
        columns = paretoFront.get(0);
        paretoFront.remove(0);

        for (int i = 0; i < paretoFront.size(); i++) {
            String[] record = paretoFront.get(i);

            int numIters = Integer.parseInt(record[getColumnIndex(columns, "numIters")]);
            int numFactors = Integer.parseInt(record[getColumnIndex(columns, "numFactors")]);
            double learningRate = Double.parseDouble(record[getColumnIndex(columns, "learningRate")]);
            double regularization = Double.parseDouble(record[getColumnIndex(columns, "regularization")]);

            DirMF dirmf = new DirMF(datamodel, numFactors, numIters, learningRate, regularization, scores, SEED);
            dirmf.fit();

            String seriesName = "DirMF_" + i;

            maePlot.addSeries(seriesName);
            coveragePlot.addSeries(seriesName);
            accuracyPlot.addSeries(seriesName);
            mapPlot.addSeries(seriesName);

            for (double rel : RELIABILITIES) {
                double mae = new MAE(dirmf, rel).getScore();
                maePlot.setValue(seriesName, rel, 1-mae/maxDiff);

                double coverage = new Coverage(dirmf, rel).getScore();
                coveragePlot.setValue(seriesName, rel, coverage);

                double accuracy = new ClassificationAccuracy(dirmf, rel).getScore();
                accuracyPlot.setValue(seriesName, rel, accuracy);

                double map = new MAP(dirmf, NUMBER_OF_RECOMMENDATIONS, relevantScore, rel).getScore();
                mapPlot.setValue(seriesName, rel, map);
            }
        }


        // PMF

        paretoFront = getParetoFront("pmf");
        columns = paretoFront.get(0);
        paretoFront.remove(0);

        String[] best = paretoFront.get(0);

        int numIters = Integer.parseInt(best[getColumnIndex(columns, "numIters")]);
        int numFactors = Integer.parseInt(best[getColumnIndex(columns, "numFactors")]);
        double lambda = Double.parseDouble(best[getColumnIndex(columns, "lambda")]);
        double gamma = Double.parseDouble(best[getColumnIndex(columns, "gamma")]);

        PMF pmf = new PMF(datamodel, numFactors, numIters, lambda, gamma, SEED);
        pmf.fit();

        String seriesName = "PMF";

        maePlot.addSeries(seriesName);
        coveragePlot.addSeries(seriesName);
        mapPlot.addSeries(seriesName);

        for (double rel : RELIABILITIES) {
            double mae = new MAE(pmf, rel).getScore();
            maePlot.setValue(seriesName, rel, 1-mae/maxDiff);

            double coverage = new Coverage(pmf, rel).getScore();
            coveragePlot.setValue(seriesName, rel, coverage);

            double map = new MAP(pmf, NUMBER_OF_RECOMMENDATIONS, relevantScore, rel).getScore();
            mapPlot.setValue(seriesName, rel, map);
        }


        // MLP

        paretoFront = getParetoFront("mlp");
        columns = paretoFront.get(0);
        paretoFront.remove(0);

        best = paretoFront.get(0);

        int numEpochs = Integer.parseInt(best[getColumnIndex(columns, "numEpochs")]);
        numFactors = Integer.parseInt(best[getColumnIndex(columns, "numFactors")]);
        double learningRate = Double.parseDouble(best[getColumnIndex(columns, "learningRate")]);

        MLP mlp = new MLP(datamodel, numFactors, numEpochs, learningRate, SEED);
        mlp.fit();

        seriesName = "MLP";

        maePlot.addSeries(seriesName);
        coveragePlot.addSeries(seriesName);
        mapPlot.addSeries(seriesName);

        for (double rel : RELIABILITIES) {
            double mae = new MAE(mlp, rel).getScore();
            maePlot.setValue(seriesName, rel, 1-mae/maxDiff);

            double coverage = new Coverage(mlp, rel).getScore();
            coveragePlot.setValue(seriesName, rel, coverage);

            double map = new MAP(mlp, NUMBER_OF_RECOMMENDATIONS, relevantScore, rel).getScore();
            mapPlot.setValue(seriesName, rel, map);
        }


        // Export results

        maePlot.exportData("results/test-split/" + DATASET + "/mae.csv");
        coveragePlot.exportData("results/test-split/" + DATASET + "/coverage.csv");
        accuracyPlot.exportData("results/test-split/" + DATASET + "/accuracy.csv");
        mapPlot.exportData("results/test-split/" + DATASET + "/map.csv");
    }

    private static List<String[]> getParetoFront (String method) throws Exception {
        String fileName = "results/gridsearch/" + DATASET + "/" + method + ".csv";
        CSVReader csvReader = new CSVReaderBuilder(new FileReader(fileName)).build();

        List<String[]> records = csvReader.readAll();
        csvReader.close();

        String[] columns = records.get(0);
        records.remove(0);

        List<String[]> paretoFront = new ArrayList<>();
        paretoFront.add(columns);

        for (int i = 0; i < records.size(); i++) {
            if (isInParetoFront(records, i, columns)) {
                paretoFront.add(records.get(i));
            }
        }

        return paretoFront;
    }

    private static int getColumnIndex (String[] columns, String columnName) {
        for (int i = 0; i < columns.length; i++) {
            if (columns[i].equals(columnName)) {
                return i;
            }
        }
        return -1;
    }

    private static boolean isInParetoFront (List<String[]> records, int index, String[] columns) {
        int maeIndex = getColumnIndex(columns, "cummulativemae_avg");
        int coverageIndex = getColumnIndex(columns, "cummulativecoverage_avg");

        double mae = Double.parseDouble(records.get(index)[maeIndex]);
        double coverage = Double.parseDouble(records.get(index)[coverageIndex]);

        if (Double.isNaN(mae) || Double.isNaN((coverage))) {
            return false;
        }

        for (int i = 0; i < records.size(); i++) {
            String[] record = records.get(i);

            if (i != index) {
                double recordMae = Double.parseDouble(record[maeIndex]);
                double recordCoverage = Double.parseDouble(record[coverageIndex]);

                if (!Double.isNaN(recordMae) && !Double.isNaN(recordCoverage)) {
                    if ((recordMae > mae && recordCoverage >= coverage) || (recordMae >= mae && recordCoverage > coverage)) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
}

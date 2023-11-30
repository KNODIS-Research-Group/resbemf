package es.upm.etsisi.knodis.resbemf.recommender;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.util.Maths;
import es.upm.etsisi.cf4j.util.process.Parallelizer;
import es.upm.etsisi.cf4j.util.process.Partible;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Load predictions performed by Berg, R. V. D., Kipf, T. N., & Welling, M. (2017). Graph convolutional
 * matrix completion. arXiv preprint arXiv:1706.02263.
 *
 * Experiments to obtain predictions has been runned using https://github.com/riannevdberg/gc-mc
 */
public class GCMC extends ProbabilistcRecommender {

    protected final String predsFile;

    protected final Map<String, double[]> testPredictions;

    protected final double[] scores;


    public GCMC(DataModel datamodel, String predsFile, double[] scores) {
        super(datamodel);
        this.testPredictions = new HashMap<>();
        this.predsFile = predsFile;
        this.scores = scores;
    }


    @Override
    public void fit() {
        System.out.println("\nFitting " + this.toString());

        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(this.predsFile));
            String header = reader.readLine(); // ignore

            String line = reader.readLine();
            while (line != null) {
                String[] split = line.split(",");

                int u = Integer.parseInt(split[0]);
                int i = Integer.parseInt(split[1]);

                double[] probs = new double[this.scores.length];
                for (int s = 0; s < probs.length; s++) {
                    probs[s] = Double.parseDouble(split[s+3]);
                }

                this.testPredictions.put("u" + u + "i" + i, probs);

                line = reader.readLine();
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public double predict(int userIndex, int itemIndex) {
        double[] probs = this.testPredictions.get("u" + userIndex + "i" + itemIndex);

        double max = probs[0];
        int index = 0;

        for (int s = 1; s < this.scores.length; s++) {
            if (max < probs[s]) {
                max = probs[s];
                index = s;
            }
        }

        return this.scores[index];
    }

    @Override
    public double mean(int userIndex, int itemIndex) {
        double[] probs = this.testPredictions.get("u" + userIndex + "i" + itemIndex);
        double mean = 0;
        for (int s = 0; s < this.scores.length; s++) {
            mean += this.scores[s] * probs[s];
        }
        return mean;
    }

    public double predictProba(int userIndex, int itemIndex) {
        double prediction = this.predict(userIndex, itemIndex);

        int s = 0;
        while (this.scores[s] != prediction) {
            s++;
        }

        double[] probs = this.testPredictions.get("u" + userIndex + "i" + itemIndex);
        return probs[s];
    }

    public double[] getProbs(int userIndex, int itemIndex) {
        double[] probs = this.testPredictions.get("u" + userIndex + "i" + itemIndex);
        return probs;
    }


    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("GCMC");
        return str.toString();
    }
}

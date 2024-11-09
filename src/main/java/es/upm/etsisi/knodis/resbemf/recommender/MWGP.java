package es.upm.etsisi.knodis.resbemf.recommender;

import es.upm.etsisi.cf4j.data.DataModel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Load predictions performed by Berg, R. V. D., Kipf, T. N., & Welling, M. (2017). Graph convolutional
 * matrix completion. arXiv preprint arXiv:1706.02263.
 *
 * Experiments to obtain predictions has been runned using https://github.com/riannevdberg/gc-mc
 */
public class MWGP extends ProbabilistcRecommender {

    protected final String predsFile;

    protected final Map<String, Double> testPredictions;

    public MWGP(DataModel datamodel, String predsFile) {
        super(datamodel);
        this.testPredictions = new HashMap<>();
        this.predsFile = predsFile;
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
                double pred = Double.parseDouble(split[3]);

                this.testPredictions.put("u" + u + "i" + i, pred);

                line = reader.readLine();
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public double predict(int userIndex, int itemIndex) {
        double pred = this.testPredictions.get("u" + userIndex + "i" + itemIndex);
        return pred;
    }

    @Override
    public double mean(int userIndex, int itemIndex) {
        return predict(userIndex, itemIndex);
    }

    public double predictProba(int userIndex, int itemIndex) {
        return 1;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("MWGP");
        return str.toString();
    }
}

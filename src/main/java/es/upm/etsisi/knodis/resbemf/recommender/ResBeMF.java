package es.upm.etsisi.knodis.resbemf.recommender;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.util.Maths;
import es.upm.etsisi.cf4j.util.process.Parallelizer;
import es.upm.etsisi.cf4j.util.process.Partible;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public class ResBeMF extends ProbabilistcRecommender {

    /** Number of latent factors */
    private final int numFactors;

    /** Number of iterations */
    private final int numIters;

    /** Learning rate */
    private final double learningRate;

    /** Regularization parameter */
    private final double regularization;

    /** Plausible rating scores **/
    private final double[] scores;

    /** Users factors **/
    private final double[][][] P;

    /** Items factors **/
    private final double[][][] Q;

    /**
     * Model constructor from a Map containing the model's hyper-parameters values. Map object must
     * contains the following keys:
     *
     * <ul>
     *   <li><b>numFactors</b>: int value with the number of latent factors.</li>
     *   <li><b>numIters:</b>: int value with the number of iterations.</li>
     *   <li><b>learningRate</b>: double value with the learning rate hyper-parameter.</li>
     *   <li><b>regularization</b>: double value with the regularization hyper-parameter.</li>
     *   recklessness
     *   recklessnessType
     *   <li><b>scores</b>: plausible rating scores.</li>
     *   <li><b><em>seed</em></b> (optional): random seed for random numbers generation. If missing,
     *       random value is used.
     * </ul>
     *
     * @param datamodel DataModel instance
     * @param params Model's hyper-parameters values
     */
    public ResBeMF(DataModel datamodel, Map<String, Object> params) {
        this(
                datamodel,
                (int) params.get("numFactors"),
                (int) params.get("numIters"),
                (double) params.get("learningRate"),
                (double) params.get("regularization"),
                (double[]) params.get("scores"),
                params.containsKey("seed") ? (long) params.get("seed") : System.currentTimeMillis()
        );
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of latent factors
     * @param numIters Number of iterations
     * @param learningRate Learning rate
     * @param regularization Regularization
     * @param scores Plausible rating scores
     * @param seed Seed for random numbers generation
     */
    public ResBeMF(DataModel datamodel, int numFactors, int numIters, double learningRate, double regularization, double[] scores, long seed) {
        super(datamodel);

        this.numFactors = numFactors;
        this.numIters = numIters;
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.scores = scores;

        Random rand = new Random(seed);

        this.P = new double[datamodel.getNumberOfUsers()][scores.length][numFactors];
        for (int u = 0; u < datamodel.getNumberOfUsers(); u++) {
            for (int s = 0; s < scores.length; s++) {
                for (int k = 0; k < numFactors; k++) {
                    this.P[u][s][k] = rand.nextDouble();
                }
            }
        }

        this.Q = new double[datamodel.getNumberOfItems()][scores.length][numFactors];
        for (int i = 0; i < datamodel.getNumberOfItems(); i++) {
            for (int r = 0; r < scores.length; r++) {
                for (int k = 0; k < numFactors; k++) {
                    this.Q[i][r][k] = rand.nextDouble();
                }
            }
        }
    }

    /**
     * Get the number of factors of the model
     *
     * @return Number of factors
     */
    public int getNumFactors() {
        return numFactors;
    }

    /**
     * Get the number of iterations
     *
     * @return Number of iterations
     */
    public int getNumIters() {
        return numIters;
    }

    /**
     * Get the learning rate parameter of the model
     *
     * @return Learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Get the regularization parameter of the model
     *
     * @return Regularization
     */
    public double getRegularization() {
        return regularization;
    }

    /**
     * Get the plausible rating scores
     *
     * @return Plausible rating scores
     */
    public double[] getScores() {
        return scores;
    }

    @Override
    public void fit() {
        System.out.println("\nFitting " + this.toString());

        for (int iter = 1; iter <= this.numIters; iter++) {
            Parallelizer.exec(this.datamodel.getUsers(), new UpdateUsersFactors());
            Parallelizer.exec(this.datamodel.getItems(), new UpdateItemsFactors());

            if ((iter % 10) == 0) System.out.print(".");
            if ((iter % 100) == 0) System.out.println(iter + " iterations");
        }
    }

    @Override
    public double predict(int userIndex, int itemIndex) {
        double max = this.softmax(userIndex, itemIndex, 0);
        int index = 0;

        for (int s = 1; s < this.scores.length; s++) {
            double prob = this.softmax(userIndex, itemIndex, s);
            if (max < prob) {
                max = prob;
                index = s;
            }
        }

        return this.scores[index];
    }

    @Override
    public double mean(int userIndex, int itemIndex) {
        double mean = 0;
        for (int s = 0; s < this.scores.length; s++) {
            double prob = this.softmax(userIndex, itemIndex, s);
            mean += this.scores[s] * prob;
        }
        return mean;
    }


    private double softmax(int userIndex, int itemIndex, int s) {
        double exp = Math.exp(Maths.dotProduct(this.P[userIndex][s], this.Q[itemIndex][s]));

        double sum = 0;
        for (int i = 0; i < this.scores.length; i++) {
            sum += Math.exp(Maths.dotProduct(this.P[userIndex][i], this.Q[itemIndex][i]));
        }

        return exp / sum;
    }

    /**
     * Computes a prediction probability
     *
     * @param userIndex Index of the user in the array of Users of the DataModel instance
     * @param itemIndex Index of the item in the array of Items of the DataModel instance
     * @return Prediction probability
     */
    public double predictProba(int userIndex, int itemIndex) {
        double prediction = this.predict(userIndex, itemIndex);

        int s = 0;
        while (this.scores[s] != prediction) {
            s++;
        }

        return this.softmax(userIndex, itemIndex, s);
    }

    public double[] getProbs(int userIndex, int itemIndex) {
        double[] probs = new double[scores.length];
        for (int s = 0; s < probs.length; s++) {
            probs[s] = softmax(userIndex, itemIndex, s);
        }
        return probs;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("ResBeMF(")
                .append("numFactors=").append(this.numFactors)
                .append("; ")
                .append("numIters=").append(this.numIters)
                .append("; ")
                .append("learningRate=").append(this.learningRate)
                .append("; ")
                .append("regularization=").append(this.regularization)
                .append("; ")
                .append("scores=").append(Arrays.toString(this.scores))
                .append(")");
        return str.toString();
    }

    /**
     * Auxiliary inner class to parallelize user factors computation
     */
    private class UpdateUsersFactors implements Partible<User> {

        @Override
        public void beforeRun() { }

        @Override
        public void run(User user) {
            int userIndex = user.getUserIndex();

            double[][] gradient = new double[scores.length][numFactors];

            for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                int itemIndex = user.getItemAt(pos);
                double rating = user.getRatingAt(pos);

                double[] softmax = new double[scores.length];
                for (int s = 0; s < scores.length; s++) {
                    softmax[s] = softmax(userIndex, itemIndex, s);
                }

                int s0 = Arrays.binarySearch(scores, rating);

                for (int s = 0; s < scores.length; s++) {
                    for (int f = 0; f < numFactors; f++) {
                        if (s == s0) {
                            gradient[s][f] += learningRate * (1 - softmax[s0]) * Q[itemIndex][s][f];
                        } else {
                            gradient[s][f] -= learningRate * softmax[s0] * Q[itemIndex][s][f];
                        }

                        gradient[s][f] -= learningRate * regularization * P[userIndex][s][f];
                    }
                }
            }

            for (int s = 0; s < scores.length; s++) {
                for (int f = 0; f < numFactors; f++) {
                    P[userIndex][s][f] +=  gradient[s][f];
                }
            }
        }

        @Override
        public void afterRun() { }
    }

    /**
     * Auxiliary inner class to parallelize item factors computation
     */
    private class UpdateItemsFactors implements Partible<Item> {

        @Override
        public void beforeRun() { }

        @Override
        public void run(Item item) {
            int itemIndex = item.getItemIndex();


            double[][] gradient = new double[scores.length][numFactors];

            for (int pos = 0; pos < item.getNumberOfRatings(); pos++) {
                int userIndex = item.getUserAt(pos);
                double rating = item.getRatingAt(pos);

                double[] softmax = new double[scores.length];
                for (int s = 0; s < scores.length; s++) {
                    softmax[s] = softmax(userIndex, itemIndex, s);
                }

                int s0 = Arrays.binarySearch(scores, rating);

                for (int s = 0; s < scores.length; s++) {
                    for (int f = 0; f < numFactors; f++) {
                        if (s == s0) {
                            gradient[s][f] += learningRate * (1 - softmax[s0]) * P[userIndex][s][f];
                        } else {
                            gradient[s][f] -= learningRate * softmax[s0] * P[userIndex][s][f];
                        }

                        gradient[s][f] -= learningRate * regularization * Q[itemIndex][s][f];
                    }
                }
            }

            for (int s = 0; s < scores.length; s++) {
                for (int f = 0; f < numFactors; f++) {
                    Q[itemIndex][s][f] += gradient[s][f];
                }
            }
        }

        @Override
        public void afterRun() { }
    }
}

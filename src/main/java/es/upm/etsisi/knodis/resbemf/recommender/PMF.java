package es.upm.etsisi.knodis.resbemf.recommender;

import es.upm.etsisi.cf4j.util.process.Parallelizer;
import es.upm.etsisi.cf4j.util.process.Partible;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.util.Maths;

import java.util.Map;
import java.util.Random;

/**
 * Implements Mnih, A., &amp; Salakhutdinov, R. R. (2008). Probabilistic matrix factorization. In
 * Advances in neural information processing systems (pp. 1257-1264).
 */
public class PMF extends ProbabilistcRecommender {

    protected static final double DEFAULT_GAMMA = 0.01;
    protected static final double DEFAULT_LAMBDA = 0.05;

    /** User factors */
    protected final double[][] p;

    /** Item factors */
    protected final double[][] q;

    /** Learning rate */
    protected final double gamma;

    /** Regularization parameter */
    protected final double lambda;

    /** Number of latent factors */
    protected final int numFactors;

    /** Number of iterations */
    protected final int numIters;

    /**
     * Model constructor from a Map containing the model's hyper-parameters values. Map object must
     * contains the following keys:
     *
     * <ul>
     *   <li><b>numFactors</b>: int value with the number of latent factors.
     *   <li><b>numIters:</b>: int value with the number of iterations.
     *   <li><b><em>gamma</em></b> (optional): double value with the learning rate hyper-parameter. If
     *       missing, it is set to 0.01.
     *   <li><b><em>lambda</em></b> (optional): double value with the regularization hyper-parameter.
     *       If missing, it is set to 0.05.
     *   <li><b><em>seed</em></b> (optional): random seed for random numbers generation. If missing,
     *       random value is used.
     * </ul>
     *
     * @param datamodel DataModel instance
     * @param params Model's hyper-parameters values
     */
    public PMF(DataModel datamodel, Map<String, Object> params) {
        this(
                datamodel,
                (int) params.get("numFactors"),
                (int) params.get("numIters"),
                params.containsKey("lambda") ? (double) params.get("lambda") : DEFAULT_LAMBDA,
                params.containsKey("gamma") ? (double) params.get("gamma") : DEFAULT_GAMMA,
                params.containsKey("seed") ? (long) params.get("seed") : System.currentTimeMillis());
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     */
    public PMF(DataModel datamodel, int numFactors, int numIters) {
        this(datamodel, numFactors, numIters, DEFAULT_LAMBDA);
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param seed Seed for random numbers generation
     */
    public PMF(DataModel datamodel, int numFactors, int numIters, long seed) {
        this(datamodel, numFactors, numIters, DEFAULT_LAMBDA, DEFAULT_GAMMA, seed);
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     */
    public PMF(DataModel datamodel, int numFactors, int numIters, double lambda) {
        this(datamodel, numFactors, numIters, lambda, DEFAULT_GAMMA, System.currentTimeMillis());
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     * @param seed Seed for random numbers generation
     */
    public PMF(DataModel datamodel, int numFactors, int numIters, double lambda, long seed) {
        this(datamodel, numFactors, numIters, lambda, DEFAULT_GAMMA, seed);
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     * @param gamma Learning rate parameter
     * @param seed Seed for random numbers generation
     */
    public PMF(
            DataModel datamodel, int numFactors, int numIters, double lambda, double gamma, long seed) {
        super(datamodel);

        this.numFactors = numFactors;
        this.numIters = numIters;
        this.lambda = lambda;
        this.gamma = gamma;

        Random rand = new Random(seed);

        // Users initialization
        this.p = new double[datamodel.getNumberOfUsers()][numFactors];
        for (int u = 0; u < datamodel.getNumberOfUsers(); u++) {
            for (int k = 0; k < numFactors; k++) {
                this.p[u][k] = rand.nextDouble() * 2 - 1;
            }
        }

        // Items initialization
        this.q = new double[datamodel.getNumberOfItems()][numFactors];
        for (int i = 0; i < datamodel.getNumberOfItems(); i++) {
            for (int k = 0; k < numFactors; k++) {
                this.q[i][k] = rand.nextDouble() * 2 - 1;
            }
        }
    }

    /**
     * Get the number of factors of the model
     *
     * @return Number of factors
     */
    public int getNumFactors() {
        return this.numFactors;
    }

    /**
     * Get the number of iterations
     *
     * @return Number of iterations
     */
    public int getNumIters() {
        return this.numIters;
    }

    /**
     * Get the regularization parameter of the model
     *
     * @return Lambda
     */
    public double getLambda() {
        return this.lambda;
    }

    /**
     * Get the learning rate parameter of the model
     *
     * @return Gamma
     */
    public double getGamma() {
        return this.gamma;
    }

    /**
     * Get the latent factors vector of a user (pu)
     *
     * @param userIndex User index
     * @return Latent factors vector
     */
    public double[] getUserFactors(int userIndex) {
        return this.p[userIndex];
    }

    /**
     * Get the latent factors vector of an item (qi)
     *
     * @param itemIndex Item index
     * @return Latent factors vector
     */
    public double[] getItemFactors(int itemIndex) {
        return this.q[itemIndex];
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
        return Maths.dotProduct(this.p[userIndex], this.q[itemIndex]);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("PMF(")
                .append("numFactors=").append(this.numFactors)
                .append("; ")
                .append("numIters=").append(this.numIters)
                .append("; ")
                .append("gamma=").append(this.gamma)
                .append("; ")
                .append("lambda=").append(this.lambda)
                .append(")");
        return str.toString();
    }

    @Override
    public double predictProba(int userIndex, int itemIndex) {
        return 1;
    }

    /** Auxiliary inner class to parallelize user factors computation */
    private class UpdateUsersFactors implements Partible<User> {

        @Override
        public void beforeRun() {}

        @Override
        public void run(User user) {
            int userIndex = user.getUserIndex();
            for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                int itemIndex = user.getItemAt(pos);
                double error = user.getRatingAt(pos) - predict(userIndex, itemIndex);
                for (int k = 0; k < numFactors; k++) {
                    p[userIndex][k] += gamma * (error * q[itemIndex][k] - lambda * p[userIndex][k]);
                }
            }
        }

        @Override
        public void afterRun() {}
    }

    /** Auxiliary inner class to parallelize item factors computation */
    private class UpdateItemsFactors implements Partible<Item> {

        @Override
        public void beforeRun() {}

        @Override
        public void run(Item item) {
            int itemIndex = item.getItemIndex();
            for (int pos = 0; pos < item.getNumberOfRatings(); pos++) {
                int userIndex = item.getUserAt(pos);
                double error = item.getRatingAt(pos) - predict(userIndex, itemIndex);
                for (int k = 0; k < numFactors; k++) {
                    q[itemIndex][k] += gamma * (error * p[userIndex][k] - lambda * q[itemIndex][k]);
                }
            }
        }

        @Override
        public void afterRun() {}
    }
}

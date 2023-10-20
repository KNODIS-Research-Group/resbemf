package es.upm.etsisi.knodis.resbemf.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.util.Search;
import es.upm.etsisi.knodis.resbemf.recommender.ProbabilistcRecommender;

public class NDCG extends QualityMeasure {

    private final int numberOfRecommendations;
    private final ProbabilistcRecommender recommender;
    private final double threshold;

    public NDCG(ProbabilistcRecommender recommender, int numberOfRecommendations, double threshold) {
        super(recommender);
        this.recommender = recommender;
        this.numberOfRecommendations = numberOfRecommendations;
        this.threshold = threshold;
    }

    @Override
    protected double getScore(TestUser testUser, double[] predictions) {

        int userIndex = testUser.getTestUserIndex();

        double[] reliablePredictions = new double[predictions.length];

        for (int pos = 0; pos < testUser.getNumberOfTestRatings(); pos++) {
            int testItemIndex = testUser.getTestItemAt(pos);
            TestItem testItem = recommender.getDataModel().getTestItem(testItemIndex);
            int itemIndex = testItem.getTestItemIndex();

            double prob = recommender.predictProba(userIndex, itemIndex);
            reliablePredictions[pos] = (prob >= threshold) ? predictions[pos] : Double.NaN;
        }

        // Compute DCG

        int[] recommendations = Search.findTopN(reliablePredictions, this.numberOfRecommendations);

        double dcg = dataCalculation(testUser,recommendations);

        // Compute IDCG

        double[] testRatings = new double[testUser.getNumberOfTestRatings()];
        for (int pos = 0; pos < testRatings.length; pos++) {
            testRatings[pos] = testUser.getTestRatingAt(pos);
        }

        int[] idealRecommendations = Search.findTopN(testRatings, this.numberOfRecommendations);

        double idcg = dataCalculation(testUser,idealRecommendations);

        if (idcg == 0) return Double.NEGATIVE_INFINITY;
        // Compute NDCG
        return dcg / idcg;
    }

    protected double dataCalculation(TestUser testUser, int[] elements){
        double result = 0d;

        for (int i = 0; i < elements.length; i++) {
            int pos = elements[i];
            if (pos == -1) break;

            double rating = testUser.getTestRatingAt(pos);
            result += (Math.pow(2, rating) - 1) / (Math.log(i + 2) / Math.log(2));
        }

        return result;
    }
}

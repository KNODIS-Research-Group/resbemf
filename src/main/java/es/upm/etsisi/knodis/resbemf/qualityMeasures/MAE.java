package es.upm.etsisi.knodis.resbemf.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.knodis.resbemf.recommender.ProbabilistcRecommender;

public class MAE extends QualityMeasure {

    private ProbabilistcRecommender recommender;
    private double threshold;

    public MAE(ProbabilistcRecommender recommender, double threshold) {
        super(recommender);
        this.recommender = recommender;
        this.threshold = threshold;
    }

    @Override
    public double getScore(TestUser testUser, double[] predictions) {
        int userIndex = testUser.getTestUserIndex();

        double sum = 0d;
        int count = 0;

        for (int pos = 0; pos < testUser.getNumberOfTestRatings(); pos++) {
            int testItemIndex = testUser.getTestItemAt(pos);
            TestItem testItem = recommender.getDataModel().getTestItem(testItemIndex);
            int itemIndex = testItem.getTestItemIndex();

            double prob = recommender.predictProba(userIndex, itemIndex);
            if (prob >= threshold) {
                double prediction = predictions[pos];
                if (!Double.isNaN(prediction)) {
                    double rating = testUser.getTestRatingAt(pos);
                    sum += Math.abs(prediction - rating);
                    count++;
                }
            }
        }

        return (count == 0) ? Double.NaN : (sum / count);
    }
}
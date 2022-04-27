package es.upm.etsisi.knodis.resbemf.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.knodis.resbemf.recommender.ProbabilistcRecommender;

public class CummulativeMAE extends QualityMeasure {

    private ProbabilistcRecommender recommender;
    private int evaluatedPoints;

    public CummulativeMAE(Recommender recommender) {
        this(recommender, 20);
    }

    public CummulativeMAE(Recommender recommender, int evaluatedPoints) {
        super(recommender);
        this.recommender = (ProbabilistcRecommender) recommender;
        this.evaluatedPoints = evaluatedPoints;
    }

    @Override
    public double getScore() {

        double max = this.recommender.getDataModel().getMaxRating();
        double min = this.recommender.getDataModel().getMinRating();

        double sum = 0;

        for (int i = 0; i < evaluatedPoints; i++) {
            double threshold = (double) i / (this.evaluatedPoints - 1);

            double mae = new MAE(this.recommender, threshold).getScore();
            if (!Double.isNaN(mae)) {
                double accuracy = 1.0 - mae / (max - min);
                sum += (evaluatedPoints - i) * accuracy;
            }
        }

        double score = sum / ((evaluatedPoints + 1.0) * evaluatedPoints / 2.0);
        return score;
    }

    @Override
    public double getScore(TestUser testUser, double[] predictions) {
        return 0;
    }
}

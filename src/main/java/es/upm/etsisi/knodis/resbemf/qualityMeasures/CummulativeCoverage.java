package es.upm.etsisi.knodis.resbemf.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.knodis.resbemf.recommender.ProbabilistcRecommender;

public class CummulativeCoverage extends QualityMeasure {

    private ProbabilistcRecommender recommender;
    private int evaluatedPoints;

    public CummulativeCoverage(Recommender recommender) {
        this(recommender, 20);
    }

    public CummulativeCoverage(Recommender recommender, int evaluatedPoints) {
        super(recommender);
        this.recommender = (ProbabilistcRecommender) recommender;
        this.evaluatedPoints = evaluatedPoints;
    }

    @Override
    public double getScore() {

        double sum = 0;

        for (int i = 0; i < evaluatedPoints; i++) {
            double threshold = (double) i / (this.evaluatedPoints - 1);

            double coverage = new Coverage(this.recommender, threshold).getScore();
            sum += (evaluatedPoints - i) * coverage;
        }

        double score = sum / ((evaluatedPoints + 1.0) * evaluatedPoints / 2.0);
        return score;
    }

    @Override
    public double getScore(TestUser testUser, double[] predictions) {
        return 0;
    }
}

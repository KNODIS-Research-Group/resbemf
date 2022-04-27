package es.upm.etsisi.knodis.resbemf;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.knodis.resbemf.qualityMeasures.*;
import es.upm.etsisi.knodis.resbemf.recommender.ResBeMF;

public class Test {



    public static void main (String[] args) throws Exception {
        DataModel ml100k = BenchmarkDataModels.MovieLens100K();

        ResBeMF resbemf = new ResBeMF(ml100k, 4, 75, 0.002, 0.05, new double[]{1,2,3,4,5}, 4815162342L);
        resbemf.fit();

        System.out.println(new MAE(resbemf, 0.4).getScore());


    }
}

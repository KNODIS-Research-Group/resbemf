package es.upm.etsisi.knodis.resbemf;

import es.upm.etsisi.cf4j.data.*;

import java.io.FileWriter;
import java.io.PrintWriter;

public class ExportDataset {

    private static final String DATASET = "ml10m";
    private static final String SEP = ";";

    public static void main (String[] args) throws Exception {
        DataModel datamodel = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        } else if (DATASET.equals("ml10m")) {
            datamodel = BenchmarkDataModels.MovieLens10M();
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
        } else if (DATASET.equals("anime")) {
            datamodel = BenchmarkDataModels.MyAnimeList();
        }

        PrintWriter trainCsv = new PrintWriter(new FileWriter(DATASET +"-train.csv"));
        trainCsv.println("user" + SEP + "item" + SEP + "rating");

        for (int userIndex = 0; userIndex < datamodel.getNumberOfUsers(); userIndex++) {
            User user = datamodel.getUser(userIndex);
            for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                int itemIndex = user.getItemAt(pos);
                double rating = user.getRatingAt(pos);
                trainCsv.println(userIndex + SEP + itemIndex + SEP + rating);
            }
        }

        trainCsv.close();

        PrintWriter testCsv = new PrintWriter(new FileWriter(DATASET +"-test.csv"));
        testCsv.println("user" + SEP + "item" + SEP + "rating");

        for (int testUserIndex = 0; testUserIndex < datamodel.getNumberOfTestUsers(); testUserIndex++) {
            TestUser testUser = datamodel.getTestUser(testUserIndex);
            int userIndex = testUser.getUserIndex();
            for (int pos = 0; pos < testUser.getNumberOfTestRatings(); pos++) {
                int testItemIndex = testUser.getTestItemAt(pos);
                TestItem testItem = datamodel.getTestItem(testItemIndex);
                int itemIndex = testItem.getItemIndex();
                double rating = testUser.getTestRatingAt(pos);
                testCsv.println(userIndex + SEP + itemIndex + SEP + rating);
            }
        }

        testCsv.close();
    }
}

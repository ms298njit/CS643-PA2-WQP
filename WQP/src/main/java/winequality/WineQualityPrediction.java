package winequality;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {

    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.ERROR);

        if (args.length < 2) {
            System.err.println("Missing arguments! 1st Argument = Test Dataset, 2nd Argument = Trained ML Model Path");
            System.exit(1);
        }

        final String testDataset = args[0];
        final String mlTrainedMODEL = args[1];

        SparkSession sprkSn = new SparkSession.Builder()
                .appName("Wine Quality Prediction").getOrCreate();

        MLmodelTrainer mlMdlTrnr = new MLmodelTrainer();

        // Load model
        PipelineModel pplnMdl = PipelineModel.load(mlTrainedMODEL);

        // read and transform test data in vector format
        Dataset<Row> tstDtSt = mlMdlTrnr.readAndTransformDataSet(sprkSn, testDataset);

        // add to the model predicted data
        Dataset<Row> prdctdDtSt = pplnMdl.transform(tstDtSt);
        prdctdDtSt.show();

        // evaluate performance using absolute mean error
        mlMdlTrnr.evltPrfrmncMLmdl(prdctdDtSt);
    }
}

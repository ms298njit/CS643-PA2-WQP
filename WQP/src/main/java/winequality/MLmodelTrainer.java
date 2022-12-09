package winequality;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.HashMap;

public class MLmodelTrainer {

    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.ERROR);

        if (args.length < 3) {
            System.err.println("Missing arguments! 1st Argument = Training Dataset, 2nd Argument = Validation Dataset, 3rd Argument = Trained ML Model Path");
            System.exit(1);
        }

        final String trainingDataset = args[0];
        final String validationDataset = args[1];
        final String mlTrainedMODEL = args[2];


        MLmodelTrainer mlMdlTrnr = new MLmodelTrainer();
        SparkSession sprkSn = new SparkSession.Builder()
                .appName("Wine Quality Model").getOrCreate();

        Dataset<Row> bldDtst = mlMdlTrnr.readAndTransformDataSet(sprkSn, trainingDataset);

        //use linear regression model
        LinearRegression lnrRgrsn = new LinearRegression().setMaxIter(20)
                .setRegParam(0).setFeaturesCol("features").setLabelCol("quality");

        //create pipeline and ML model
        Pipeline ppln = new Pipeline().setStages(new PipelineStage[]{lnrRgrsn});
        PipelineModel pplnMdl = ppln.fit(bldDtst);

        //validate model and optimize performance
        Dataset<Row> vldtnDtst = mlMdlTrnr.readAndTransformDataSet(sprkSn, validationDataset);

        //predict wine quality
        Dataset<Row> prdctns = pplnMdl.transform(vldtnDtst);
        prdctns.show();

        //evaluate ML model performance
        mlMdlTrnr.evltPrfrmncMLmdl(prdctns);

        //save ML trained model
        try{
            pplnMdl.write().overwrite().save(mlTrainedMODEL);
        }catch (IOException e){
            System.out.println("Error(s) detected when writing model to the disk. +"+e.getMessage());
        }

    }

    //evaluate prediction performance of ML model
    public void evltPrfrmncMLmdl(Dataset<Row> tstDtst){
        RegressionEvaluator regressionEvaluator = new RegressionEvaluator()
                .setLabelCol("quality").setPredictionCol("prediction").setMetricName("mae");
        double absoluteMeanError = regressionEvaluator.evaluate(tstDtst);
        System.out.println("Mean absolute error:" +absoluteMeanError);
    }

    
    //build dataset
    public Dataset<Row> readAndTransformDataSet(SparkSession sprk, String flNm){

        HashMap<String, String> options = new HashMap<>();
        options.put("delimiter", ";");
        options.put("inferSchema", "true");
        options.put("header", "true");

        // read the csv into memory
        Dataset<Row> rowDataset = sprk.read().options(options).csv(flNm);

        //data cleansing
        Dataset<Row> clnDtSt = rowDataset.dropDuplicates();

        //collect feature columns (Independent Data)
        Dataset<Row> ftrClmns = clnDtSt.select("fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol");

        //aggregate feature column into single
        VectorAssembler vctrAssmblr = new VectorAssembler().setInputCols(ftrClmns.columns()).setOutputCol("features");

        // convert dataset into vector type
        Dataset<Row> fnlDtSt = vctrAssmblr.transform(clnDtSt).select("features", "quality").cache();

        return fnlDtSt;
    }

}

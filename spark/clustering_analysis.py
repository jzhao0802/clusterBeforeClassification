from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from imspaeva import BinaryClassificationEvaluatorWithPrecisionAtRecall
import numpy
import os
import time
import datetime
from imspacv import CrossValidatorWithStratificationID

# get the predicted probability in Vector
def getitem(i):
    def getitem_(v):
        return v.array.item(i)
    return udf(getitem_, DoubleType())

def main():
    # user to specify: hyper-params
    n_eval_folds = 5
    n_cv_folds = 5    
    
    grid_n_trees = [20, 30]
    grid_depth = [3, 4]
        
    # desired_recalls = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    desired_recalls = [0.05,0.10]
    
    
    
    # user to specify : seed in Random Forest model
    iseed = 42
    # user to specify: input data location
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/clean_data/"
    pos_file = "pos.csv"
    neg_file ="neg.csv"
    ss_file = "ss.csv"
    #reading in the data from S3
    spark = SparkSession.builder.appName(__file__).getOrCreate()
    org_pos_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + pos_file)
    org_neg_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + neg_file)    
    org_ss_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path +ss_file)
    
    
    # user to specify: original column names for predictors and output in data
    orgOutputCol = "label"
    matchCol = "matched_positive_id"
    nonFeatureCols = ["matched_positive_id", "label", "patid"]
    orgPredictorCols = [x for x in data.columns if x not in nonFeatureCols]
    # sanity check 
    if type(data.select(orgOutputCol).schema.fields[0].dataType) not in (DoubleType, IntegerType):
        raise TypeError("The output column is not of type integer or double. ")
    data = data.withColumn(orgOutputCol, data[orgOutputCol].cast("double"))
    # user to specify: the collective column name for all predictors
    collectivePredictorCol = "features"
    # user to specify: the column name for prediction
    predictionCol = "probability"
    # user to specify: the output location on s3
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/Hui/template_test/results/" + st + "/"
    # user to specify the output location on master
    resultDir_master = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"

    
    
    
    

    # convert to ml-compatible format
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    posFeatureAssembledData = assembler.transform(org_pos_data)\
        .select(nonFeatureCols, collectivePredictorCol)
    posFeatureAssembledData.cache()
    negFeatureAssembledData = assembler.transform(org_neg_data)\
        .select(nonFeatureCols, collectivePredictorCol)
    negFeatureAssembledData.cache()
    #
    evalIDCol = "evalFoldID"
    cvIDCol = "cvFoldID"
    pos_neg_data = posFeatureAssembledData.union(negFeatureAssembledData)
    pos_neg_data_with_eval_ids = AppendDataMatchingFoldIDs(pos_neg_data, n_eval_folds, matchCol, foldCol=evalIDCol)
    
    
    ssFeatureAssembledData = assembler.transform(org_ss_data)\
        .select(nonFeatureCols, collectivePredictorCol)
    ssFeatureAssembledData.cache()
    
    
    # the model (pipeline)
    rf = RandomForestClassifier(featuresCol = collectivePredictorCol,
                                labelCol = orgOutputCol, seed=iseed)
    Need to update the evaluator 
    evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(\
        rawPredictionCol=predictionCol,
        labelCol=orgOutputCol,
        metricName="precisionAtGivenRecall",
        metricParams={"recallValue":0.05}\
    )
    paramGrid = ParamGridBuilder()\
            .addGrid(rf.numTrees, grid_n_trees)\
            .addGrid(rf.maxDepth, grid_depth)\
            .build()

    # cross-evaluation
    predictionsAllData = None

    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3, 0o777)
    if not os.path.exists(resultDir_master):
        os.makedirs(resultDir_master, 0o777)
    os.chmod(resultDir_master, 0o777)

    for iFold in range(n_eval_folds):
        
        
        condition = featureAssembledData[outerFoldCol] == iFold
        leftoutFold = featureAssembledData.filter(condition)
        trainFolds = featureAssembledData.filter(~condition).drop(evalIDCol)
        trainDataWithCVFoldID = AppendDataMatchingFoldIDs(trainFolds, n_cv_folds, matchCol, foldCol=cvIDCol)

        validator = CrossValidatorWithStratificationID(\
                        estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        stratifyCol=cvIDCol\
                    )
        cvModel = validator.fit(trainDataWithCVFoldID)
        
        
        #
        ## test data
        testData = ssFeatureAssembledData\
            .join(leftoutFold, matchCol, how="left_outer")\
            .union(leftoutFold.drop("evalIDCol"))
        
        predictions = cvModel.transform(testData)

        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.unionAll(predictions)
        else:
            predictionsAllData = predictions

        # save the metrics for all hyper-parameter sets in cv
        cvMetrics = cvModel.avgMetrics
        cvMetricsFileName = resultDir_s3 + "cvMetricsFold" + str(iFold)
        cvMetrics.coalesce(4).write.csv(cvMetricsFileName, header="true")

        # save the hyper-parameters of the best model
        bestParams = validator.getBestModelParams()
        with open(resultDir_master + "bestParamsFold" + str(iFold) + ".txt",
                  "w") as fileBestParams:
            fileBestParams.write(str(bestParams))
        os.chmod(resultDir_master + "bestParamsFold" + str(iFold) + ".txt", 0o777)
        # save importance score of the best model
        with open(resultDir_master + "importanceScoreFold" + str(iFold) + ".txt",
                  "w") as filecvCoef:
            for id in range(len(orgPredictorCols)):
                filecvCoef.write("{0} : {1}".format(orgPredictorCols[id], cvModel.bestModel.featureImportances[id]))
                filecvCoef.write("\n")
        os.chmod(resultDir_master + "importanceScoreFold" + str(iFold) + ".txt", 0o777)

    # save all predictions
    predictionsFileName = resultDir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol,
                              getitem(1)(predictionCol).alias('prob_1'))\
        .write.csv(predictionsFileName, header="true")
    # metrics of predictions on the entire dataset
    metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": x}} for x in desired_recalls]
    metricValues = evaluator\
        .evaluateWithSeveralMetrics(dataset, metricSets = metricSets)\
        .write.csv(resultDir_s3+"metricValuesEntireData.csv", header="true")
    spark.stop()

if __name__ == "__main__":
    main()
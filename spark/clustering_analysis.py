from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
import numpy
import os
import time
import datetime
from stratification import AppendDataMatchingFoldIDs
from imspacv import CrossValidatorWithStratificationID
from imspaeva import BinaryClassificationEvaluatorWithPrecisionAtRecall

# get the predicted probability in Vector
def getitem(i):
    def getitem_(v):
        return v.array.item(i)
    return udf(getitem_, DoubleType())

def save_analysis_info(path, file_name, **kwargs):
    with open(path + file_name, "w") as file:
        for key, value in kwargs.iteritems():         
            file.write(key + ": " + str(value) + "\n")
        os.chmod(path + file_name, 0o777)
    
def main():

    # user to specify: hyper-params
    n_eval_folds = 3
    n_cv_folds = 3  
    
    grid_n_trees = [200]
    grid_depth = [5]
    minInstancesPerNode = [100]
    featureSubsetStrategy = ["onethird"]
        
    # desired_recalls = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    desired_recalls = [0.05]
    
    
    
    # user to specify : seed in Random Forest model
    seed = 42
    # user to specify: input data location
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/smaller_data/"
    pos_file = "pos_70.0pct.csv"
    neg_file = "neg_70.0pct.csv"
    ss_file = "ss_70.0pct.csv"
    # data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/smaller_data/"
    # pos_file = "pos_50.csv"
    # neg_file = "neg_50.csv"
    # ss_file = "ss_50.csv"
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
    # sanity check 
    if type(org_pos_data.select(orgOutputCol).schema.fields[0].dataType) not in (DoubleType, IntegerType):
        raise TypeError("The output column is not of type integer or double. ")
    org_pos_data = org_pos_data.withColumn(orgOutputCol, org_pos_data[orgOutputCol].cast("double"))
    orgPredictorCols = [x for x in org_pos_data.columns if x not in nonFeatureCols]    
    if type(org_neg_data.select(orgOutputCol).schema.fields[0].dataType) not in (DoubleType, IntegerType):
        raise TypeError("The output column is not of type integer or double. ")
    org_neg_data = org_neg_data.withColumn(orgOutputCol, org_neg_data[orgOutputCol].cast("double"))
    if type(org_ss_data.select(orgOutputCol).schema.fields[0].dataType) not in (DoubleType, IntegerType):
        raise TypeError("The output column is not of type integer or double. ")
    org_ss_data = org_ss_data.withColumn(orgOutputCol, org_ss_data[orgOutputCol].cast("double"))
    # user to specify: the collective column name for all predictors
    collectivePredictorCol = "features"
    # user to specify: the column name for prediction
    predictionCol = "probability"
    # user to specify: the output location on s3
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    result_dir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/" + st + "/"
    # user to specify the output location on master
    result_dir_master = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"
    if not os.path.exists(result_dir_s3):
        os.makedirs(result_dir_s3, 0o777)
    if not os.path.exists(result_dir_master):
        os.makedirs(result_dir_master, 0o777)
    
    save_analysis_info(\
        result_dir_master, 
        "analysis_info.txt", 
        n_eval_folds=n_eval_folds,
        n_cv_folds=n_cv_folds,
        grid_n_trees=grid_n_trees,
        grid_depth=grid_depth,
        desired_recalls=desired_recalls,
        seed=seed,
        data_path=data_path,
        pos_file=pos_file,
        neg_file=neg_file,
        ss_file=ss_file,
        result_dir_s3=result_dir_s3,
        result_dir_master=result_dir_master
        )
    
    

    # convert to ml-compatible format
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    posFeatureAssembledData = assembler.transform(org_pos_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    # posFeatureAssembledData.cache()
    negFeatureAssembledData = assembler.transform(org_neg_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    # negFeatureAssembledData.cache()
    #
    evalIDCol = "evalFoldID"
    cvIDCol = "cvFoldID"
    pos_neg_data = posFeatureAssembledData.union(negFeatureAssembledData)
    pos_neg_data_with_eval_ids = AppendDataMatchingFoldIDs(pos_neg_data, n_eval_folds, matchCol, foldCol=evalIDCol)
    
    
    ssFeatureAssembledData = assembler.transform(org_ss_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    # ssFeatureAssembledData.cache()
    
    
    # the model (pipeline)
    rf = RandomForestClassifier(featuresCol = collectivePredictorCol,
                                labelCol = orgOutputCol, seed=seed)
    evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(\
        rawPredictionCol=predictionCol,
        labelCol=orgOutputCol,
        metricName="precisionAtGivenRecall",
        metricParams={"recallValue":0.05}\
    )
    paramGrid = ParamGridBuilder()\
            .addGrid(rf.numTrees, grid_n_trees)\
            .addGrid(rf.maxDepth, grid_depth)\
            .addGrid(rf.minInstancesPerNode, minInstancesPerNode)\
            .addGrid(rf.featureSubsetStrategy, featureSubsetStrategy)\
            .build()

    # cross-evaluation
    predictionsAllData = None

    for iFold in range(n_eval_folds):
        
        
        condition = pos_neg_data_with_eval_ids[evalIDCol] == iFold
        leftoutFold = pos_neg_data_with_eval_ids.filter(condition)
        trainFolds = pos_neg_data_with_eval_ids.filter(~condition).drop(evalIDCol)
        trainDataWithCVFoldID = AppendDataMatchingFoldIDs(trainFolds, n_cv_folds, matchCol, foldCol=cvIDCol)
        
        validator = CrossValidatorWithStratificationID(\
                        estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        stratifyCol=cvIDCol\
                    )
        trainDataWithCVFoldID.cache()
        cvModel = validator.fit(trainDataWithCVFoldID)
        trainDataWithCVFoldID.unpersist()
        
        
        #
        ## test data
        testData = ssFeatureAssembledData\
            .join(leftoutFold.select(matchCol), matchCol)\
            .union(leftoutFold.drop(evalIDCol))
        testData.cache()
        predictions = cvModel.transform(testData)
        predictions.write.csv(result_dir_s3+"predictions_fold_"+str(iFold)+".csv", header="true")
        
        # if predictionsAllData is not None:
            # predictionsAllData = predictionsAllData.union(predictions)
        # else:
            # predictionsAllData = predictions
        # predictionsAllData.cache()

        # save the metrics for all hyper-parameter sets in cv
        cvMetrics = cvModel.avgMetrics
        cvMetricsFileName = result_dir_s3 + "cvMetricsFold" + str(iFold)
        cvMetrics.coalesce(4).write.csv(cvMetricsFileName, header="true")

        # save the hyper-parameters of the best model
        bestParams = validator.getBestModelParams()
        with open(result_dir_master + "bestParamsFold" + str(iFold) + ".txt",
                  "w") as fileBestParams:
            fileBestParams.write(str(bestParams))
        os.chmod(result_dir_master + "bestParamsFold" + str(iFold) + ".txt", 0o777)
        # save importance score of the best model
        with open(result_dir_master + "importanceScoreFold" + str(iFold) + ".txt",
                  "w") as filecvCoef:
            for id in range(len(orgPredictorCols)):
                filecvCoef.write("{0} : {1}".format(orgPredictorCols[id], cvModel.bestModel.featureImportances[id]))
                filecvCoef.write("\n")
        os.chmod(result_dir_master + "importanceScoreFold" + str(iFold) + ".txt", 0o777)
        
        testData.unpersist()

    # # save all predictions
    # predictionsFileName = result_dir_s3 + "predictionsAllData"
    # predictionsAllData.select(orgOutputCol,
                              # getitem(1)(predictionCol).alias('prob_1'))\
        # .write.csv(predictionsFileName, header="true")
    # # metrics of predictions on the entire dataset
    # metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": x}} for x in desired_recalls]
    # metricValues = evaluator\
        # .evaluateWithSeveralMetrics(predictionsAllData, metricSets = metricSets)
    # with open(result_dir_master + "metricValuesEntireData.csv", "w") as file:
        # for elem in metricValues:
            # key = elem.keys()[0]
            # value = elem.values()[0]
            # file.write(key + "," + str(value) + "\n")
    # os.chmod(result_dir_master + "metricValuesEntireData.csv", 0o777)
    # predictionsAllData.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
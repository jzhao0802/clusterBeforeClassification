from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
import pyspark.sql.functions as F
import pyspark
import numpy
import os
import sys
import math
from stratification import AppendDataMatchingFoldIDs
from imspacv import CrossValidatorWithStratificationID
from imspaeva import BinaryClassificationEvaluatorWithPrecisionAtRecall

# get the predicted probability in Vector
def getitem(i):
    def getitem_(v):
        return v.array.item(i)
    return udf(getitem_, DoubleType())


def save_analysis_info(path, file_name, configs):
    with open(path + file_name, "w") as file:
        for key, value in configs.items():         
            file.write(key + ": " + str(value) + "\n")
        os.chmod(path + file_name, 0o777)


def save_metrics(file_name, dfMetrics): 
    with open(file_name, "w") as file:
        for elem in dfMetrics:
            key = elem.keys()[0]
            value = elem.values()[0]
            file.write(key + "," + str(value) + "\n")
    os.chmod(file_name, 0o777)

    
def main(result_dir_master, result_dir_s3):
    
    CON_CONFIGS = {}
    CON_CONFIGS["result_dir_master"] = result_dir_master
    CON_CONFIGS["result_dir_s3"] = result_dir_s3

    #
    ## user to specify: hyper-params
    
    # classification
    CON_CONFIGS["n_eval_folds"] = 3
    CON_CONFIGS["n_cv_folds"] = 53
    
    CON_CONFIGS["lambdas"] = [0.1, 1]
    CON_CONFIGS["alphas"] = [0]
        
    # CON_CONFIGS["desired_recalls"] = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    CON_CONFIGS["desired_recalls"] = [0.05,0.10]
    
    
    
    #
    ## read data and some meta studff
    
    
    # user to specify : seed in Random Forest model
    CON_CONFIGS["seed"] = 42
    CON_CONFIGS["data_path"] = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/smaller_data/"
    CON_CONFIGS["pos_file"] = "pos_1.0pct.csv"
    CON_CONFIGS["neg_file"] = "neg_1.0pct_ratio_5.csv"
    CON_CONFIGS["ss_file"] = "ss_1.0pct_ratio_10.csv"
    #reading in the data from S3
    spark = SparkSession.builder.appName(os.path.basename(__file__)).getOrCreate()
    org_pos_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(CON_CONFIGS["data_path"] + CON_CONFIGS["pos_file"])
    org_neg_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(CON_CONFIGS["data_path"] + CON_CONFIGS["neg_file"])\
        .select(org_pos_data.columns)
    org_ss_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(CON_CONFIGS["data_path"] +CON_CONFIGS["ss_file"])\
        .select(org_pos_data.columns)
    
    
    # user to specify: original column names for predictors and output in data
    orgOutputCol = "label"
    matchCol = "matched_positive_id"
    patIDCol = "patid"
    nonFeatureCols = [matchCol, orgOutputCol, patIDCol]
    orgPredictorCols = ["PATIENT_AGE", "LOOKBACK_DAYS", "LVL3_CHRN_ISCH_HD_FLAG", "LVL3_ABN_CHST_XRAY_FLAG"]
    org_pos_data = org_pos_data.select(nonFeatureCols + orgPredictorCols)
    org_neg_data = org_neg_data.select(nonFeatureCols + orgPredictorCols)
    org_ss_data = org_ss_data.select(nonFeatureCols + orgPredictorCols)
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
    
    CON_CONFIGS["orgPredictorCols"] = orgPredictorCols
    CON_CONFIGS["n_predictors_classification"] = len(orgPredictorCols)
    CON_CONFIGS["n_rows_pos"] = org_pos_data.count()
    CON_CONFIGS["n_rows_neg"] = org_neg_data.count()
    CON_CONFIGS["n_rows_ss"] = org_ss_data.count()
    save_analysis_info(\
        result_dir_master, 
        "analysis_info.txt", 
        CON_CONFIGS
        )
    
    
    
    
    
    
    
    # convert to ml-compatible format
    assembler = VectorAssembler(inputCols=orgPredictorCols, outputCol=collectivePredictorCol)
    posFeatureAssembledData = assembler.transform(org_pos_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    negFeatureAssembledData = assembler.transform(org_neg_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    #
    evalIDCol = "evalFoldID"
    cvIDCol = "cvFoldID"
    pos_neg_data = posFeatureAssembledData.union(negFeatureAssembledData)
    pos_neg_data_with_eval_ids = AppendDataMatchingFoldIDs(pos_neg_data, CON_CONFIGS["n_eval_folds"], matchCol, foldCol=evalIDCol)
    
    
    # the model (pipeline)
    classifier_spec = LogisticRegression(maxIter=1e5, featuresCol = collectivePredictorCol,
                            labelCol = orgOutputCol, standardization = True)
    evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(\
        rawPredictionCol=predictionCol,
        labelCol=orgOutputCol,
        metricName="precisionAtGivenRecall",
        metricParams={"recallValue":0.05}\
    )
    paramGrid = ParamGridBuilder()\
               .addGrid(classifier_spec.regParam, CON_CONFIGS["lambdas"])\
               .addGrid(classifier_spec.elasticNetParam, CON_CONFIGS["alphas"])\
               .build()

    # cross-evaluation
    predictionsAllData = None
    
    metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": x}} for x in CON_CONFIGS["desired_recalls"]]
    
    filename_loop_info = result_dir_master + "loop_info.txt"
    file_loop_info = open(filename_loop_info, "w")    
    
    for iFold in range(CON_CONFIGS["n_eval_folds"]):
        
        
        condition = pos_neg_data_with_eval_ids[evalIDCol] == iFold
        leftoutFold = pos_neg_data_with_eval_ids.filter(condition).drop(evalIDCol)
        trainFolds = pos_neg_data_with_eval_ids.filter(~condition).drop(evalIDCol)
        
        file_loop_info.write("####################################################################\n\n".format(iFold))
        file_loop_info.write("iFold: {}\n\n".format(iFold))
        file_loop_info.write("n_rows of leftoutFold: {}\n".format(leftoutFold.count()))
        file_loop_info.write("n_rows of trainFolds: {}\n".format(trainFolds.count()))        
        
        trainDataWithCVFoldID = AppendDataMatchingFoldIDs(trainFolds, CON_CONFIGS["n_cv_folds"], matchCol, foldCol=cvIDCol)
        trainDataWithCVFoldID.coalesce(int(trainFolds.rdd.getNumPartitions()))
        
        #
        ## train the classifier     
        

        validator = CrossValidatorWithStratificationID(\
                        estimator=classifier_spec,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        stratifyCol=cvIDCol\
                    )
        cvModel = validator.fit(trainDataWithCVFoldID)
        
        predictionsOneFold = cvModel\
            .transform(leftoutFold)\
            .select(nonFeatureCols + [collectivePredictorCol, predictionCol])
        
        metricValuesOneFold = evaluator\
            .evaluateWithSeveralMetrics(predictionsOneFold, metricSets = metricSets)            
        file_name_metrics_one_fold = result_dir_master + "metrics_fold_" + str(iFold) + "_.csv"
        save_metrics(file_name_metrics_one_fold, metricValuesOneFold)
        predictionsOneFold\
            .select(orgOutputCol, getitem(1)(predictionCol).alias('prob_1'))\
            .write.csv(result_dir_s3 + "predictions_fold_" + str(iFold) + ".csv")
        predictionsOneFold.persist(pyspark.StorageLevel(True, False, False, False, 1))

        # save the metrics for all hyper-parameter sets in cv
        cvMetrics = cvModel.avgMetrics
        cvMetricsFileName = result_dir_s3 + "cvMetrics_fold_" + str(iFold)
        cvMetrics.coalesce(4).write.csv(cvMetricsFileName, header="true")

        # save the hyper-parameters of the best model
        
        bestParams = validator.getBestModelParams()
        file_best_params = result_dir_master + "bestParams_fold_" + str(iFold) + ".txt"
        with open(file_best_params, "w") as fileBestParams:
            fileBestParams.write(str(bestParams))
        os.chmod(file_best_params, 0o777)
        
        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.union(predictionsOneFold)
        else:
            predictionsAllData = predictionsOneFold            

    # save all predictions
    predictionsFileName = result_dir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol,
                              getitem(1)(predictionCol).alias('prob_1'))\
        .write.csv(predictionsFileName, header="true")
    # metrics of predictions on the entire dataset
    metricValues = evaluator\
        .evaluateWithSeveralMetrics(predictionsAllData, metricSets = metricSets)
    save_metrics(result_dir_master + "metricValuesEntireData.csv", metricValues)
    
    file_loop_info.close()
    os.chmod(file_loop_info, 0o777)
    
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
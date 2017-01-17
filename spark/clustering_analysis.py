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

def vec_to_numpy_pair_dist(p1_vec, p2_numpy_array):
    return math.sqrt(p1_vec.squared_distance(p2_numpy_array))

def compute_and_append_dist_to_numpy_array_point(row, featureCol, target_point, distCol): 
    source_point = row[featureCol]
    dist = math.sqrt(source_point.squared_distance(target_point))
    elems = row.asDict()
    if distCol in elems.keys():
        raise ValueError("distCol already exists in the input data point. Please choose a new name.")
    
    elems[distCol] = dist
    
    return Row(**elems)
    
def compute_and_append_in_cluster_dist(row, featureCol, clusterCol, centres, distCol):
    cluster_id = row[clusterCol]
    centre = centres[cluster_id] 
    return compute_and_append_dist_to_numpy_array_point(row, featureCol, centre, distCol)
        
def clustering(data_4_clustering_assembled, clustering_obj, clusterFeatureCol, clusterCol, distCol): 
    data_4_clustering_assembled.cache()
    cluster_model = clustering_obj.fit(data_4_clustering_assembled)
    cluster_result = cluster_model.transform(data_4_clustering_assembled)
    centres = cluster_model.clusterCenters()
    result_with_dist = cluster_result.rdd\
        .map(lambda x: compute_and_append_in_cluster_dist(x, clusterFeatureCol, clusterCol, centres, distCol))\
        .toDF()
    data_4_clustering_assembled.unpersist()
    return (cluster_model, result_with_dist)

def append_id(elem, new_elem_name):
    row_elems = elem[0].asDict()
    if new_elem_name in row_elems.keys():
        raise ValueError(new_elem_name + "already exists in the original data frame. ")
    row_elems[new_elem_name] = elem[1]
    
    return Row(**row_elems)
    
def select_certain_pct_ids_per_positive_closest_to_cluster_centre(assembled_data_4_clustering, clusterFeatureCol, centre, similar_pct, idCol, matchCol):
    distCol = "_tmp_dist"
    nPoses = assembled_data_4_clustering.select(matchCol).distinct().count()
    num_to_retain = round(assembled_data_4_clustering.count() / float(nPoses) * similar_pct)
    dist_df = assembled_data_4_clustering.rdd\
        .map(lambda x: compute_and_append_dist_to_numpy_array_point(x, clusterFeatureCol, centre, distCol))\
        .toDF()
    dist_df.registerTempTable("dist_table")
    ids = assembled_data_4_clustering.sql_ctx.sql(\
        "SELECT " + idCol + " FROM (SELECT *, row_number() OVER(PARTITION BY " + matchCol + " ORDER BY " + distCol + ") AS tmp_rank FROM dist_table) WHERE tmp_rank <=" + str(num_to_retain)
    )
        
    SparkSession.builder.getOrCreate().catalog.dropTempView("dist_table")
    
    return ids

def select_certain_pct_overall_ids_closest_to_cluster_centre(assembled_data_4_clustering, clusterFeatureCol, centre, similar_pct, idCol):
    distCol = "_tmp_dist"
    num_to_retain = round(assembled_data_4_clustering.count() * similar_pct)
    ids = assembled_data_4_clustering.rdd\
        .map(lambda x: compute_and_append_dist_to_numpy_array_point(x, clusterFeatureCol, centre, distCol))\
        .toDF()\
        .sort(distCol)\
        .rdd.zipWithIndex()\
        .map(lambda x: append_id(x, "_tmp_id"))\
        .toDF()\
        .filter(F.col("_tmp_id") < num_to_retain)\
        .select(idCol)
        
    return ids
    
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
    CON_CONFIGS["n_eval_folds"] = 5
    CON_CONFIGS["n_cv_folds"] = 5  
    
    CON_CONFIGS["lambdas"] = list(10.0 ** numpy.arange(-2, 2, 1.0))
    CON_CONFIGS["alphas"] = list(numpy.linspace(0, 1, 3))
        
    CON_CONFIGS["desired_recalls"] = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    # CON_CONFIGS["desired_recalls"] = [0.05,0.10]
    
    
    
    #
    ## read data and some meta studff
    
    
    # user to specify : seed in Random Forest model
    CON_CONFIGS["seed"] = 42
    CON_CONFIGS["data_path"] = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/smaller_data/"
    CON_CONFIGS["pos_file"] = "pos_70.0pct.csv"
    CON_CONFIGS["neg_file"] = "neg_70.0pct.csv"
    CON_CONFIGS["ss_file"] = "ss_70.0pct.csv"
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
    # orgPredictorCols = ["PATIENT_AGE", "LOOKBACK_DAYS", "LVL3_CHRN_ISCH_HD_FLAG", "LVL3_ABN_CHST_XRAY_FLAG"]
    # org_pos_data = org_pos_data.select(nonFeatureCols + orgPredictorCols)
    # org_neg_data = org_neg_data.select(nonFeatureCols + orgPredictorCols)
    # org_ss_data = org_ss_data.select(nonFeatureCols + orgPredictorCols)
    # sanity check 
    if type(org_pos_data.select(orgOutputCol).schema.fields[0].dataType) not in (DoubleType, IntegerType):
        raise TypeError("The output column is not of type integer or double. ")
    org_pos_data = org_pos_data.withColumn(orgOutputCol, org_pos_data[orgOutputCol].cast("double"))
    orgPredictorCols = [x for x in org_pos_data.columns if x not in nonFeatureCols]    
    orgPredictorCols4Clustering = [x for x in orgPredictorCols if "FLAG" in x]
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
        
        metricValuesOneCluster = evaluator\
            .evaluateWithSeveralMetrics(predictionsOneFold, metricSets = metricSets)            
        file_name_metrics_one_cluster = result_dir_master + "metrics_fold_" + str(iFold) + "_.csv"
        save_metrics(file_name_metrics_one_cluster, metricValuesOneCluster)
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
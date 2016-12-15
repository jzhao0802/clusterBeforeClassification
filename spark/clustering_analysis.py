from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.clustering import KMeans
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

def vec_to_numpy_pair_dist(p1_vec, p2_numpy_array):
    return math.sqrt(p1_vec.squared_distance(p2_numpy_array))

def compute_and_append_dist_to_numpy_array_point(row, clusterFeatureCol, target_point, distCol): 
    vec1 = row[featureCol]
    dist = math.sqrt(vec1.squared_distance(target_point))
    elems = row.asDict()
    if distCol in elems.keys():
        raise ValueError("distCol already exists in the input data point. Please choose a new name.")
    
    elems[distCol] = dist
    
    return Row(**elems)
    
def compute_and_append_in_cluster_dist(row, featureCol, clusterCol, centres, distCol):
    cluster_id = row[clusterCol]
    vec2 = centres[cluster_id] 
    return compute_and_append_dist_to_numpy_array_point(row, clusterFeatureCol, vec2, distCol)
        
def clustering(data_4_clustering_assembled, clustering_obj, clusterFeatureCol, clusterCol, distCol): 
    cluster_model = clustering_obj.fit(data_4_clustering_assembled)
    cluster_result = cluster_model.transform(pos_data_4_clustering_assembled)
    centres = cluster_model.clusterCenters()
    result_with_dist = cluster_result.rdd\
        .map(lambda x: compute_and_append_in_cluster_dist(x, clusterFeatureCol, clusterCol, centres, distCol))\
        .toDF()
        
    return (cluster_model, result_with_dist)
        
def main():
    #
    ## user to specify: hyper-params
    
    # clustering
    n_clusters = 3
    dist_threshold_percentile = 0.75
    
    # classification
    n_eval_folds = 3
    n_cv_folds = 3  
    
    grid_n_trees = [20, 30]
    grid_depth = [3]
        
    # desired_recalls = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    desired_recalls = [0.05,0.10]
    
    
    
    #
    ## read data and some meta studff
    
    
    # user to specify : seed in Random Forest model
    seed = 42
    # user to specify: input data location
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/clean_data/"
    pos_file = "pos.csv"
    neg_file = "neg.csv"
    ss_file = "ss.csv"
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
    patIDCol = "patid"
    nonFeatureCols = [matchCol, orgOutputCol, patIDCol]
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
    # 
    clusterFeatureCol = "cluster_features"
    clusterCol = "cluster_id"
    # user to specify: the collective column name for all predictors
    collectivePredictorCol = "features"
    # in-cluster distance
    distCol = "dist"
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
    posFeatureAssembledData.cache()
    negFeatureAssembledData = assembler.transform(org_neg_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    negFeatureAssembledData.cache()
    #
    evalIDCol = "evalFoldID"
    cvIDCol = "cvFoldID"
    pos_neg_data = posFeatureAssembledData.union(negFeatureAssembledData)
    pos_neg_data_with_eval_ids = AppendDataMatchingFoldIDs(pos_neg_data, n_eval_folds, matchCol, foldCol=evalIDCol)
    
    
    ssFeatureAssembledData = assembler.transform(org_ss_data)\
        .select(nonFeatureCols + [collectivePredictorCol])
    ssFeatureAssembledData.cache()
    
    
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
            .build()

    # cross-evaluation
    predictionsAllData = None
    
    kmeans = KMeans(featureCol=clusterFeatureCol, predictionCol=clusterCol).setK(n_clusters)
    cluster_assembler = VectorAssembler(inputCols=orgPredictorCols4Clustering, outputCol=clusterFeatureCol)
    
        

    for iFold in range(n_eval_folds):
        
        
        condition = pos_neg_data_with_eval_ids[evalIDCol] == iFold
        leftoutFold = pos_neg_data_with_eval_ids.filter(condition)
        trainFolds = pos_neg_data_with_eval_ids.filter(~condition).drop(evalIDCol)
        
        #
        ## clustering to be done here
        
        pos_data_4_clustering = trainFolds\
            .filter(orgOutputCol==1)\
            .select(patIDCol, matchCol)\
            .join(org_pos_data, matchCol)
        pos_data_4_clustering_assembled = cluster_assembler.transform(pos_data_4_clustering)\
            .select([patIDCol, matchCol] + [collectivePredictorCol])
        cluster_model, clustered_pos = clustering(pos_data_4_clustering_assembled, kmeans, 
                                    clusterFeatureCol, clusterCol, distCol) 
        
        for i_cluster in range(n_clusters):
            
            # the positive data for training the classifier
            train_pos = clustered_pos\
                .filter(clustered_pos[clusterCol]==i_cluster)\
                .select(patIDCol)\
                .join(trainFolds, patIDCol)
            
            # select negative training data based on the clustering result
            corresponding_neg = trainFolds.filter(trainFolds[orgOutputCol]==0)
            corresponding_neg_4_clustering_assembled = cluster_assembler.transform(corresponding_neg)\
                .select([patIDCol, matchCol] + [collectivePredictorCol])
            similar_neg = corresponding_neg_4_clustering_assembled.rdd\
                .map(lambda x: compute_and_append_dist_to_numpy_array_point(x, clusterFeatureCol, cluster_model.clusterCenters()[i_cluster], distCol))\
                .toDF()
            
            
            
        
        
        
        
        Now AppendDataMatchingFoldIDs is tricky. There might not be enough negatives for some positives. 
        just try two ways. 
        
        trainDataWithCVFoldID = AppendDataMatchingFoldIDs(trainFolds, n_cv_folds, matchCol, foldCol=cvIDCol)
        
        
        
        #
        ## classification for each cluster
        
        
        
        

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
            .join(leftoutFold.select(matchCol), matchCol)\
            .union(leftoutFold.drop(evalIDCol))
        
        predictions = cvModel.transform(testData)

        if predictionsAllData is not None:
            predictionsAllData = predictionsAllData.unionAll(predictions)
        else:
            predictionsAllData = predictions

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

    # save all predictions
    predictionsFileName = result_dir_s3 + "predictionsAllData"
    predictionsAllData.select(orgOutputCol,
                              getitem(1)(predictionCol).alias('prob_1'))\
        .write.csv(predictionsFileName, header="true")
    # metrics of predictions on the entire dataset
    metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": x}} for x in desired_recalls]
    metricValues = evaluator\
        .evaluateWithSeveralMetrics(predictionsAllData, metricSets = metricSets)
    with open(result_dir_master + "metricValuesEntireData.csv", "w") as file:
        for elem in metricValues:
            key = elem.keys()[0]
            value = elem.values()[0]
            file.write(key + "," + str(value) + "\n")
    os.chmod(result_dir_master + "metricValuesEntireData.csv", 0o777)
    spark.stop()

if __name__ == "__main__":
    main()
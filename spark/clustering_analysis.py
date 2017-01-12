from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
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

def save_analysis_info(path, file_name, **kwargs):
    with open(path + file_name, "w") as file:
        for key, value in kwargs.iteritems():         
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
    cluster_model = clustering_obj.fit(data_4_clustering_assembled)
    cluster_result = cluster_model.transform(data_4_clustering_assembled)
    centres = cluster_model.clusterCenters()
    result_with_dist = cluster_result.rdd\
        .map(lambda x: compute_and_append_in_cluster_dist(x, clusterFeatureCol, clusterCol, centres, distCol))\
        .toDF()
        
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
    #
    ## user to specify: hyper-params
    
    # clustering
    n_clusters = 3
    warn_threshold_np_ratio = 5
    
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
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/BI/smaller_data/"
    pos_file = "pos_1.0pct.csv"
    neg_file = "neg_1.0pct.csv"
    ss_file = "ss_1.0pct.csv"
    #reading in the data from S3
    spark = SparkSession.builder.appName(os.path.basename(__file__)).getOrCreate()
    org_pos_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + pos_file)
    org_neg_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + neg_file)\
        .select(org_pos_data.columns)
    org_ss_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path +ss_file)\
        .select(org_pos_data.columns)
    
    
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
    orgPredictorCols4Clustering = [x for x in orgPredictorCols if "FLAG" in x]
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
        result_dir_master=result_dir_master,
        n_clusters = n_clusters,
        warn_threshold_np_ratio = warn_threshold_np_ratio
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
    pos_neg_data_with_eval_ids = AppendDataMatchingFoldIDs(pos_neg_data, n_eval_folds, matchCol, foldCol=evalIDCol)
    
    
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
    
    kmeans = KMeans(featuresCol=clusterFeatureCol, predictionCol=clusterCol).setK(n_clusters)
    cluster_assembler = VectorAssembler(inputCols=orgPredictorCols4Clustering, outputCol=clusterFeatureCol)
    
    metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": x}} for x in desired_recalls]
    

    for iFold in range(n_eval_folds):
        
        
        condition = pos_neg_data_with_eval_ids[evalIDCol] == iFold
        leftoutFold = pos_neg_data_with_eval_ids.filter(condition).drop(evalIDCol)
        trainFolds = pos_neg_data_with_eval_ids.filter(~condition).drop(evalIDCol)
        
        #
        ## clustering to be done here
        
        pos_data_4_clustering = trainFolds\
            .filter(F.col(orgOutputCol)==1)\
            .select(patIDCol)\
            .join(org_pos_data, patIDCol)
        pos_data_4_clustering_assembled = cluster_assembler.transform(pos_data_4_clustering)\
            .select([patIDCol, matchCol] + [clusterFeatureCol])
        cluster_model, clustered_pos = clustering(pos_data_4_clustering_assembled, kmeans, 
                                    clusterFeatureCol, clusterCol, distCol) 
        
        nPosesAllClusters = clustered_pos.count()
        predictionsOneFold = None
        
        for i_cluster in range(n_clusters):
            
            # the positive data for training the classifier
            train_pos = clustered_pos\
                .filter(clustered_pos[clusterCol]==i_cluster)\
                .select(patIDCol)\
                .join(trainFolds, patIDCol)
            
            posPctThisClusterVSAllClusters = float(train_pos.count()) / nPosesAllClusters
            # select negative training data based on the clustering result
            corresponding_neg = train_pos\
                .select(matchCol)\
                .join(org_neg_data, matchCol)
            corresponding_neg_4_clustering_assembled = cluster_assembler.transform(corresponding_neg)\
                .select([patIDCol, matchCol] + [clusterFeatureCol])
            similar_neg_ids = select_certain_pct_ids_per_positive_closest_to_cluster_centre(\
                corresponding_neg_4_clustering_assembled, 
                clusterFeatureCol, 
                cluster_model.clusterCenters()[i_cluster], 
                posPctThisClusterVSAllClusters, 
                patIDCol,
                matchCol
            )
            train_data = similar_neg_ids\
                .join(trainFolds, patIDCol)\
                .select(train_pos.columns)\
                .union(train_pos)
            
            trainDataWithCVFoldID = AppendDataMatchingFoldIDs(train_data, n_cv_folds, matchCol, foldCol=cvIDCol)
            # sanity check: if there are too few negatives for any positive 
            thresh_n_neg_per_fold = round(train_pos.count() / float(n_cv_folds)) * warn_threshold_np_ratio
            neg_counts_all_cv_folds = trainDataWithCVFoldID\
                .filter(F.col(orgOutputCol)==0)\
                .groupBy(cvIDCol)\
                .agg(F.count(orgOutputCol).alias("_tmp"))\
                .select("_tmp")\
                .collect()
            if any(map(lambda x: x["_tmp"] < thresh_n_neg_per_fold, neg_counts_all_cv_folds)):
                raise ValueError("Insufficient number of negative data in at least one cv fold.")
                
            
        
        
            #
            ## train the classifier     
            

            validator = CrossValidatorWithStratificationID(\
                            estimator=rf,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            stratifyCol=cvIDCol\
                        )
            cvModel = validator.fit(trainDataWithCVFoldID)
            
            
            #
            ## test data
            
            
            entireTestData = org_ss_data\
                .join(leftoutFold.filter(F.col(orgOutputCol)==1).select(matchCol), matchCol).select(org_pos_data.columns)\
                .union(org_pos_data.join(leftoutFold.select(patIDCol), patIDCol).select(org_pos_data.columns))\
                .union(org_neg_data.join(leftoutFold.select(patIDCol), patIDCol).select(org_pos_data.columns))
            entireTestDataAssembled4Clustering = cluster_assembler.transform(entireTestData)\
                    .select([patIDCol, matchCol] + [clusterFeatureCol])
            
            filteredTestData = select_certain_pct_overall_ids_closest_to_cluster_centre(\
                entireTestDataAssembled4Clustering, 
                clusterFeatureCol, 
                cluster_model.clusterCenters()[i_cluster], 
                posPctThisClusterVSAllClusters, 
                patIDCol
            ).join(entireTestData, patIDCol)
            
            filteredTestDataAssembled = assembler.transform(filteredTestData)\
                .select(nonFeatureCols + [collectivePredictorCol])       
            
            
            # testing
            
            predictions = cvModel.transform(filteredTestDataAssembled)
            metricValuesOneCluster = evaluator\
                .evaluateWithSeveralMetrics(predictions, metricSets = metricSets)            
            file_name_metrics_one_cluster = result_dir_master + "metrics_cluster_" + str(i_cluster) + "fold_" + str(iFold) + "_.csv"
            save_metrics(file_name_metrics_one_cluster, metricValuesOneCluster)
            predictions.write.csv(result_dir_s3 + "predictions_fold_" + str(iFold) + "_cluster_" + str(i_cluster) + ".csv")
            

            if predictionsOneFold is not None:
                predictionsOneFold = predictionsOneFold.union(predictions)
            else:
                predictionsOneFold = predictions
            
            # save the metrics for all hyper-parameter sets in cv
            cvMetrics = cvModel.avgMetrics
            cvMetricsFileName = result_dir_s3 + "cvMetrics_cluster_" + str(i_cluster) + "_fold_" + str(iFold)
            cvMetrics.coalesce(4).write.csv(cvMetricsFileName, header="true")

            # save the hyper-parameters of the best model
            
            bestParams = validator.getBestModelParams()
            file_best_params = result_dir_master + "bestParams_cluster_" + str(i_cluster) + "_fold_" + str(iFold) + ".txt"
            with open(file_best_params, "w") as fileBestParams:
                fileBestParams.write(str(bestParams))
            os.chmod(file_best_params, 0o777)
            
        
        # summarise all clusters from the fold
        
        metricValuesOneFold = evaluator\
            .evaluateWithSeveralMetrics(predictionsOneFold, metricSets = metricSets)            
        file_name_metrics_one_fold = result_dir_master + "metrics_fold_" + str(iFold) + "_.csv"
        save_metrics(file_name_metrics_one_fold, metricValuesOneFold)
        
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
    
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
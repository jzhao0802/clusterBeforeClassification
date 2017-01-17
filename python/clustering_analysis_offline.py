import os
import time
import datetime
import pandas
import numpy
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve


def read_dataset(path):
    all_files = os.listdir(path)
    csv_files = [f for f in all_files if (os.path.isfile(path + f) & (".csv" in f) & (os.stat(path + f).st_size != 0))]
    data_concat = None
    for file_name in csv_files:
        data_this_file = pandas.read_csv(path + file_name)
        if data_concat is None:
            data_concat = data_this_file
        else:
            data_concat = pandas.concat([data_concat, data_this_file])

    return data_concat


def save_analysis_info(path, file_name, configs):
    with open(path + file_name, "w") as file:
        for key, value in configs.items():
            file.write(key + ": " + str(value) + "\n")
        os.chmod(path + file_name, 0o777)


def append_data_matching_fold_ids(data, n_folds, match_col, fold_col):
    n_data = data.shape[0]
    n_cols = data.shape[1]
    arr = numpy.tile(numpy.arange(n_folds), numpy.ceil(n_data / float(n_folds)))[0:n_data]
    data.insert(n_cols, fold_col, arr)

    return data


def cal_dist(pd_s, np_c):
    return numpy.linalg.norm(pd_s.as_matrix() - np_c)


def select_certain_pct_ids_per_positive_closest_to_cluster_centre(data,
                                                                  feature_cols,
                                                                  centre,
                                                                  similar_pct,
                                                                  pat_id_col,
                                                                  match_col):
    dist_col = "_tmp_dist"
    nPoses = data.loc[:, match_col].unique().shape[0]
    num_to_retain = round(data.shape[0] / float(nPoses) * similar_pct)

    data_with_dist = data.insert(
        data.shape[1],
        dist_col,
        data.apply(cal_dist, axis=1, args=(centre, ))
    )

    result = data_with_dist\
        .groupby(match_col)\
        .apply(lambda x: pandas.DataFrame.sort(x.loc[:, dist_col]).iloc[0:(num_to_retain + 1), :])\
        .loc[:, pat_id_col]

    return result


def cross_validate_with_stratification_id(estimator, param_grid, evaluate_metric, evaluate_metric_params,
                                          fold_id_col, feature_cols, label_col, data_with_fold_id):
    param_iterable = ParameterGrid(param_grid)
    n_folds = data_with_fold_id.loc[:, fold_id_col].unique().shape[0]
    valid_params = estimator.get_params()

    for i_fold in range(n_folds):
        condition = data_with_fold_id.loc[:, fold_id_col] == i_fold
        test_fold = data_with_fold_id.loc[condition, :]
        train_folds = data_with_fold_id.loc[~condition, :]

        i_param_set = 0
        for params in param_iterable:
            # set the parameters
            for key, value in params.items():
                if key not in valid_params:
                    raise ValueError('Invalid parameter {} for estimator {}'.format(key, estimator))
                setattr(estimator, key, value)

            # train and evaluate the model / params
            model = estimator.fit(train_folds.loc[:, feature_cols], train_folds[:, label_col])
            preds = model.predict_prob(test_fold.loc[: feature_cols])


            i_param_set += 1


def main(result_dir):

    CON_CONFIGS = {}
    CON_CONFIGS["result_dir_master"] = result_dir
    CON_CONFIGS["result_dir_s3"] = result_dir

    #
    ## user to specify: hyper-params

    # clustering
    CON_CONFIGS["n_clusters"] = 3
    CON_CONFIGS["warn_threshold_np_ratio"] = 1

    # classification
    CON_CONFIGS["n_eval_folds"] = 3
    CON_CONFIGS["n_cv_folds"] = 3

    CON_CONFIGS["lambdas"] = [0.1, 1]
    CON_CONFIGS["alphas"] = [0, 0.5]

    # CON_CONFIGS["desired_recalls"] = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    CON_CONFIGS["desired_recalls"] = [0.05,0.10]



    #
    ## read data and some meta studff

    CON_CONFIGS["data_path"] = "C:/Work/Projects/PA_Research/ClusteringBeforeClassification/data/BI_IPF/"
    CON_CONFIGS["pos_file"] = "pos_1.0pct.csv"
    CON_CONFIGS["neg_file"] = "neg_1.0pct_ratio_5.csv"
    CON_CONFIGS["ss_file"] = "ss_1.0pct_ratio_10.csv"

    org_pos_data = read_dataset(CON_CONFIGS["data_path"] + CON_CONFIGS["pos_file"] + "/")
    org_neg_data = read_dataset(CON_CONFIGS["data_path"] + CON_CONFIGS["neg_file"] + "/")
    org_ss_data = read_dataset(CON_CONFIGS["data_path"] + CON_CONFIGS["ss_file"] + "/")





    # user to specify: original column names for predictors and output in data
    org_output_col = "label"
    match_col = "matched_positive_id"
    pat_id_col = "patid"
    non_feature_cols = [match_col, org_output_col, pat_id_col]
    org_predictor_cols_classification = ["PATIENT_AGE", "LOOKBACK_DAYS", "LVL3_CHRN_ISCH_HD_FLAG",
                                         "LVL3_ABN_CHST_XRAY_FLAG"]
    org_pos_data = org_pos_data.loc[:, non_feature_cols + org_predictor_cols_classification]
    org_neg_data = org_neg_data.loc[:, non_feature_cols + org_predictor_cols_classification]
    org_ss_data = org_ss_data.loc[:, non_feature_cols + org_predictor_cols_classification]

    org_predictor_cols_classification = [x for x in org_pos_data.columns if x not in non_feature_cols]
    org_predictor_cols_clustering = [x for x in org_predictor_cols_classification if "FLAG" in x]

    #
    cluster_col = "cluster_id"
    # user to specify: the collective column name for all predictors
    # in-cluster distance
    dist_col = "dist"
    # user to specify: the column name for prediction
    prediction_col = "probability"

    # user to specify : seed in Random Forest model
    CON_CONFIGS["seed"] = 42


    CON_CONFIGS["org_predictor_cols_classification"] = org_predictor_cols_classification
    CON_CONFIGS["org_predictor_cols_clustering"] = org_predictor_cols_clustering
    CON_CONFIGS["n_predictors_classification"] = len(org_predictor_cols_classification)
    CON_CONFIGS["n_rows_pos"] = org_pos_data.shape[0]
    CON_CONFIGS["n_rows_neg"] = org_neg_data.shape[0]
    CON_CONFIGS["n_rows_ss"] = org_ss_data.shape[0]
    save_analysis_info(result_dir, "analysis_info.txt", CON_CONFIGS)


    #
    ##

    eval_id_col = "evalFoldID"
    cv_id_col = "cvFoldID"
    pos_neg_data = pandas.concat([org_pos_data, org_neg_data])
    pos_neg_data_with_eval_ids = append_data_matching_fold_ids(pos_neg_data, CON_CONFIGS["n_eval_folds"], match_col,
                                                               foldCol=eval_id_col)

    #
    classifier_spec = SGDClassifier(loss="log", penalty="elasticnet")
    param_grid = {"alpha": CON_CONFIGS["lambdas"],
                  "l1_ratio": CON_CONFIGS["alphas"]}

    #
    ## loops

    for i_eval_fold in range(CON_CONFIGS["n_eval_folds"]):

        condition = pos_neg_data_with_eval_ids.loc[:, eval_id_col] == i_eval_fold
        leftout_fold = pos_neg_data_with_eval_ids.loc[condition, :].drop(eval_id_col)
        train_folds = pos_neg_data_with_eval_ids.loc[~condition, :].drop(eval_id_col)

        # clustering

        pos_data_4_clustering = pandas.merge(
            train_folds.loc[train_folds.loc[:, org_output_col] == 1, pat_id_col],
            org_pos_data,
            pat_id_col
        ).loc[:, [pat_id_col] + org_predictor_cols_clustering]

        kmeans = KMeans(n_clusters=CON_CONFIGS["n_clusters"]).fit(pos_data_4_clustering)
        clustered_pos = pos_data_4_clustering.insert(
            pos_data_4_clustering.shape[1],
            cluster_col,
            kmeans.predict(pos_data_4_clustering.loc[:, org_predictor_cols_clustering])
        )

        n_poses_all_clusters = clustered_pos.shape[0]
        predictions_one_fold = None

        for i_cluster in range(CON_CONFIGS["n_clusters"]):

            train_pos = clustered_pos[clustered_pos.loc[:, cluster_col]==i_cluster, pat_id_col]\
                .merge(train_folds, pat_id_col)
            pos_pct_this_cluster_vs_all_clusters = float(train_pos.count()) / n_poses_all_clusters
            corresponding_neg = train_pos.loc[:, match_col].merge(org_neg_data, match_col)
            similar_neg_ids = select_certain_pct_ids_per_positive_closest_to_cluster_centre(
                corresponding_neg,
                org_predictor_cols_clustering,
                kmeans.cluster_centers_[i_cluster, :],
                pos_pct_this_cluster_vs_all_clusters,
                pat_id_col,
                match_col
            )
            train_data = pandas.concat(
                [train_pos, similar_neg_ids.merge(train_folds, pat_id_col).loc[:, train_pos.columns]],
                axis=0
            )

            train_data_with_cv_fold_id = append_data_matching_fold_ids(
                train_data, CON_CONFIGS["n_cv_folds"],
                match_col,
                cv_id_col
            )

            print("Not standardising the data..")

            cv_model = cross_validate_with_stratification_id(
                estimator=classifier_spec,
                param_grid=param_grid,
                evaluate_metric="precisionAtGivenRecall",
                evaluate_metric_params={"recallValue":0.05},
                stratify_col=cv_id_col,
                feature_cols=org_predictor_cols_classification,
                label_col=org_output_col,
                data=train_data_with_cv_fold_id
            )

            # validator = CrossValidatorWithStratificationID(\
            #                 estimator=classifier_spec,
            #                 estimatorParamMaps=paramGrid,
            #                 evaluator=evaluator,
            #                 stratifyCol=cvIDCol\
            #             )
            # cvModel = validator.fit(trainDataWithCVFoldID)


            entireTestData = org_ss_data\
                .join(leftoutFold.filter(F.col(orgOutputCol)==1).select(matchCol), matchCol).select(org_pos_data.columns)\
                .union(org_pos_data.join(leftoutFold.select(patIDCol), patIDCol).select(org_pos_data.columns))\
                .union(org_neg_data.join(leftoutFold.select(patIDCol), patIDCol).select(org_pos_data.columns))
            entireTestDataAssembled4Clustering = cluster_assembler.transform(entireTestData)\
                    .select([patIDCol, matchCol] + [clusterFeatureCol])
            file_loop_info.write("n_rows of entireTestData: {}\n".format(entireTestData.count()))

            filteredTestData = select_certain_pct_overall_ids_closest_to_cluster_centre(\
                entireTestDataAssembled4Clustering,
                clusterFeatureCol,
                cluster_model.clusterCenters()[i_cluster],
                posPctThisClusterVSAllClusters,
                patIDCol
            ).join(entireTestData, patIDCol)

            file_loop_info.write("n_rows of filteredTestData: {}\n".format(filteredTestData.count()))

            filteredTestDataAssembled = assembler.transform(filteredTestData)\
                .select(nonFeatureCols + [collectivePredictorCol])

            # testing

            predictions = cvModel.transform(filteredTestDataAssembled)
            metricValuesOneCluster = evaluator\
                .evaluateWithSeveralMetrics(predictions, metricSets = metricSets)
            file_name_metrics_one_cluster = result_dir_master + "metrics_cluster_" + str(i_cluster) + "fold_" + str(iFold) + "_.csv"
            save_metrics(file_name_metrics_one_cluster, metricValuesOneCluster)
            predictions.write.csv(result_dir_s3 + "predictions_fold_" + str(iFold) + "_cluster_" + str(i_cluster) + ".csv")
            predictions.persist(pyspark.StorageLevel(True, False, False, False, 1))

            if predictionsOneFold is not None:
                predictionsOneFold = predictionsOneFold.union(predictions)
            else:
                predictionsOneFold = predictions

            # need to union the test data filtered away (all classified as negative)

            discarded_test_ids = entireTestData\
                .select(patIDCol)\
                .subtract(filteredTestData.select(patIDCol))
            discardedTestData = discarded_test_ids\
                .join(entireTestData, patIDCol)
            discardedTestDataAssembled = assembler.transform(discardedTestData, )\
                .select(nonFeatureCols + [collectivePredictorCol])
            predictionsDiscardedTestData = discardedTestDataAssembled\
                .withColumn(inputTrivialNegPredCols[0], F.lit(0.0))\
                .withColumn(inputTrivialNegPredCols[1], F.lit(1.0))
            predictionsDiscardedTestDataAssembled = trivial_neg_pred_assembler\
                .transform(predictionsDiscardedTestData)\
                .select(predictions.columns)

            predictionsEntireTestData = predictions.union(predictionsDiscardedTestDataAssembled)





            metricValuesOneCluster = evaluator\
                .evaluateWithSeveralMetrics(predictionsEntireTestData, metricSets = metricSets)
            file_name_metrics_one_cluster = result_dir_master + "metrics_cluster_" + str(i_cluster) + "fold_" + str(iFold) + "_.csv"
            save_metrics(file_name_metrics_one_cluster, metricValuesOneCluster)
            predictionsEntireTestData.write.csv(result_dir_s3 + "predictions_fold_" + str(iFold) + "_cluster_" + str(i_cluster) + ".csv")
            predictionsEntireTestData.persist(pyspark.StorageLevel(True, False, False, False, 1))

            if predictionsOneFold is not None:
                predictionsOneFold = predictionsOneFold.union(predictionsEntireTestData)
            else:
                predictionsOneFold = predictionsEntireTestData

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

    pass

if __name__ == "__main__":
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    main_result_dir = "../Results/" + st + "/"
    if not os.path.exists(main_result_dir):
        os.makedirs(main_result_dir, 0o777)
    main(main_result_dir)
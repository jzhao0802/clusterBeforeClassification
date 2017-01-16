import os
import time
import datetime
import pandas

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


    CON_CONFIGS["data_path"] = "F:/Lichao/work/Projects/BI_IPF/data/smaller_different_pn_proportion_data/"
    CON_CONFIGS["pos_file"] = "pos_1.0pct.csv"
    CON_CONFIGS["neg_file"] = "neg_1.0pct_ratio_5.csv"
    CON_CONFIGS["ss_file"] = "ss_1.0pct_ratio_10.csv"

    org_pos_data = read_dataset(CON_CONFIGS["data_path"] + CON_CONFIGS["pos_file"] + "/")
    org_neg_data = read_dataset(CON_CONFIGS["data_path"] + CON_CONFIGS["neg_file"] + "/")
    org_ss_data = read_dataset(CON_CONFIGS["data_path"] + CON_CONFIGS["ss_file"] + "/")





    # user to specify: original column names for predictors and output in data
    orgOutputCol = "label"
    matchCol = "matched_positive_id"
    patIDCol = "patid"
    nonFeatureCols = [matchCol, orgOutputCol, patIDCol]
    orgPredictorCols = ["PATIENT_AGE", "LOOKBACK_DAYS", "LVL3_CHRN_ISCH_HD_FLAG", "LVL3_ABN_CHST_XRAY_FLAG"]
    org_pos_data = org_pos_data.loc[:, nonFeatureCols + orgPredictorCols]
    org_neg_data = org_neg_data.loc[:, nonFeatureCols + orgPredictorCols]
    org_ss_data = org_ss_data.loc[:, nonFeatureCols + orgPredictorCols]

    org_pos_data = org_pos_data.withColumn(orgOutputCol, org_pos_data[orgOutputCol].cast("double"))
    orgPredictorCols = [x for x in org_pos_data.columns if x not in nonFeatureCols]
    orgPredictorCols4Clustering = [x for x in orgPredictorCols if "FLAG" in x]

    org_neg_data = org_neg_data.withColumn(orgOutputCol, org_neg_data[orgOutputCol].cast("double"))

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

    # user to specify : seed in Random Forest model
    CON_CONFIGS["seed"] = 42


    CON_CONFIGS["orgPredictorCols"] = orgPredictorCols
    CON_CONFIGS["orgPredictorCols4Clustering"] = orgPredictorCols4Clustering
    save_analysis_info(\
        result_dir_master,
        "analysis_info.txt",
        CON_CONFIGS
        )

    pass

if __name__ == "__main__":
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    main_result_dir = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"
    if not os.path.exists(main_result_dir):
        os.makedirs(main_result_dir, 0o777)
    main(main_result_dir)
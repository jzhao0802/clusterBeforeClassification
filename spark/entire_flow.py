from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os
import time
import datetime

# def dimension_reduction(org_data):
    

# def record_experiment_info(result_dir_master):
    # pass

# To add simulation ID or fold ID
def addID(dataset, number, npar, name):
    nPoses = dataset.count()
    npFoldIDsPos = np.array(list(range(number)) * np.ceil(float(nPoses) / number))
    # select the actual numbers of FoldIds matching the count of positive data points
    npFoldIDs = npFoldIDsPos[:nPoses]
    # Shuffle the foldIDs to give randomness
    np.random.shuffle(npFoldIDs)
    rddFoldIDs = sc.parallelize(npFoldIDs, npar).map(int)
    dfDataWithIndex = dataset.rdd.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "orgData")
    dfNewKeyWithIndex = rddFoldIDs.zipWithIndex() \
        .toDF() \
        .withColumnRenamed("_1", "key")
    dfJoined = dfDataWithIndex.join(dfNewKeyWithIndex, "_2") \
        .select('orgData.matched_positive_id', 'key') \
        .withColumnRenamed('key', name) \
        .coalesce(npar)
    return dfJoined
    
def main(data_path, pos_file, neg_file, ss_file, num_sim):
    #
    #
    # only use part (e.g., half) of the data    
    #
    #
    
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    result_dir_master = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"
    
    record_experiment_info(result_dir_master)
    
    
    
    org_pos_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + pos_file)
    org_neg_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + neg_file)
    org_ss_data = spark.read.option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_path + ss_file)
    pos_assembled = assembler(org_pos_data, inc_var, 'features')
    neg_assembled = assembler(org_neg_data, inc_var, 'features')
    ss_assembled = assembler(org_ss_data, inc_var, 'features')

    #union All positive and negative data as dataset
    dataset = pos_assembled.union(neg_assembled)

    #create a dataframe which has 2 column, 1 is patient ID, other one is simid
    patid_pos = pos_assembled.select('matched_positive_id')
    patsim = addID(patid_pos, num_sim, par, 'simid')
    patsim.cache()
    
    sim_result_ls = map(sim_function(isim, patsim=patsim, dataset=dataset, ss_ori=ss_ori, fold=fold), 
                        range(num_sim))

    patsim.unpersist()
    
    # for i_eval_fold in range(n_eval_folds):
    
        # train_val_data = bla
        # test_data = bla
    
        # clustering_data = dimension_reduction(train_val_data)
        
        # n_clusters = 3
        # clusters = clustering(clustering_data, n_clusters)
        
        # for cluster_id in range(n_clusters):
            # cluster_info = clusters.filter(F.col("prediction")==cluster_id).select("prim_key")
            
            # pos_model_data_this_cluster = train_val_data.join(\
                # cluster_info,
                # train_val_data["prim_key"] == cluster_info["prim_key"],
                # how="right_outer"
            # ).drop(cluster_info["prim_key"])
            
            # neg_model_data_this_cluster = bla
            # model_data_this_cluster = pos_model_data_this_cluster.union(neg_model_data_this_cluster)
            # two_stage_model = two_stage_model_train(model_data_this_cluster)
            # two_stage_model_apply(two_stage_model, test_data)

if __name__ == "__main__":
    main_data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/20161205_162617"
    main_pos_file = "pos.csv"
    main_neg_file = "neg.csv"
    main_ss_file = "ss.csv"
    main_num_sim = 3
    main(main_data_path, main_pos_file, main_neg_file, main_ss_file, main_num_sim)
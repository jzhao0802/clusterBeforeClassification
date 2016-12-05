import pyspark.sql.functions as F

# def dimension_reduction(org_data):
    

# def record_experiment_info(result_dir_master):
    # pass

def main():
    #
    #
    # only use part (e.g., half) of the data    
    #
    #
    
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    result_dir_master = "/home/lichao.wang/code/lichao/test/Results/" + st + "/"
    
    record_experiment_info(result_dir_master)
    
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
    main()
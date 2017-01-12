from pyspark.sql import SparkSession
import os
import time
import datetime

def main():
    
    # nPosesToRetain = 50
    retain_ratios = [0.001, 0.01]
    
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

    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    result_dir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/" + st + "/"    
    
    for retain_ratio in retain_ratios:    
        keyCol = "matched_positive_id"
        retained_pos = org_pos_data.sample(False, retain_ratio, seed=1).coalesce(1)
        if retained_pos.count() != retained_pos.select(keyCol).distinct().count():
            raise ValueError("Sampling result has duplicated matched_positive_ids.")
        retained_neg = org_neg_data.join(retained_pos.select(keyCol), keyCol).coalesce(4)
        retained_ss = org_ss_data.join(retained_pos.select(keyCol), keyCol).coalesce(10)
        
        retained_pos.write.csv(result_dir_s3+"pos_" + str(retain_ratio*100) + "pct.csv", header="true")
        retained_neg.write.csv(result_dir_s3+"neg_" + str(retain_ratio*100) + "pct.csv", header="true")
        retained_ss.write.csv(result_dir_s3+"ss_" + str(retain_ratio*100) + "pct.csv", header="true")
    
    spark.stop()
    
if __name__ == "__main__":
    main()
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
import os
import time
import datetime
import numpy

def topcoding(dataset):
    dataset2 = dataset\
        .withColumn('SYMP_CNT2', when(dataset.SYMP_CNT>7, 7).otherwise(dataset.SYMP_CNT))\
        .withColumn('COMOR_CNT2', when(dataset.COMOR_CNT > 5, 5). otherwise(dataset.COMOR_CNT))\
        .withColumn('NRCOMOR_CNT2', when(dataset.NRCOMOR_CNT>11, 11).otherwise(dataset.NRCOMOR_CNT))\
        .withColumn('PROC_CNT2', when(dataset.PROC_CNT>5, 5).otherwise(dataset.PROC_CNT))\
        .drop('SYMP_CNT').drop('COMOR_CNT').drop('NRCOMOR_CNT')\
        .drop('PROC_CNT')
    return dataset2
    
def drop_rename(dataset, file):
    if 'replace_extremes' in file:
        if 'pos' in file:
            dataset2 = dataset.drop('nonipf_patid')
            dataset3 = dataset2.withColumn("matched_positive_id", dataset2["patid"])
        else:
            dataset3 = dataset.withColumnRenamed("patid", "matched_positive_id")\
                .withColumnRenamed("nonipf_patid", "patid")
    elif 'for_orla' not in file:
        dataset3 = dataset.withColumnRenamed("matched_patient_id", "matched_positive_id")\
             .withColumnRenamed("patient_id", "patid")
    elif 'CNT_xform' in file:
        dataset3 = dataset.drop('').drop('X')\
            .withColumnRenamed("matched_patient_id", "matched_positive_id")\
             .withColumnRenamed("patient_id", "patid")
    return dataset3

# To read in the excluding variable names and generate excluded variable list
def exc_list(d_path, file):
    data = numpy.loadtxt(d_path + file ,dtype=numpy.str,delimiter=',',skiprows=0)
    var_ls = data[1:, 0].tolist()
    var_flag_ls = [x + '_FLAG' for x in var_ls]
    var_avg_ls = [x + '_AVG_RXDX' for x in var_ls]
    var_exc_ls = var_flag_ls + var_avg_ls
    return var_exc_ls
    
def main():
    data_path = "s3://emr-rwes-pa-spark-dev-datastore/BI_IPF_2016/01_data/"
    pos_file = "all_features_pos.csv"
    neg_file ="all_features_neg.csv"
    ss_file = 'all_features_score.csv'
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
    
    #top coding 4 count variables
    pos = topcoding(org_pos_data)
    neg = topcoding(org_neg_data)
    ss = topcoding(org_ss_data)
    
    #rename patid etc..
    pos = drop_rename(pos, pos_file)
    neg = drop_rename(neg, neg_file)
    ss = drop_rename(ss, ss_file)
    
    pos_col = pos.columns
    #include variable list
    common_list = ['matched_positive_id', 'label', 'patid']
    exc_var_list = exc_list("/home/lichao.wang/data/BI/", "vars_to_exclude.csv")
    #orla would like to remove "LVL3_GASTROSC_AVG_RXDX" as well on 0824
    inc_vars = [x for x in pos_col if x not in
               exc_var_list+common_list+['LVL3_GASTROSC_AVG_RXDX']]
               
    pos.select(inc_vars).write.csv(result_dir_s3+"pos.csv", header="true")
    neg.select(inc_vars).write.csv(result_dir_s3+"neg.csv", header="true")
    ss.select(inc_vars).write.csv(result_dir_s3+"ss.csv", header="true")
    
    spark.stop()
    
if __name__ == "__main__":
    main()
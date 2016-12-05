'''
this function is to implement 2 stage model with LR model
'''

# Import the packages
import sys
import os
import time
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors
import numpy as np
#import pandas as pd
import random
import csv
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import monotonicallyIncreasingId

#from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
#from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

app_name = sys.argv[1]
s3_path = "s3://emr-rwes-pa-spark-dev-datastore"
seed = 42
par = 5

######## !!!!!!!Some variables, change before running!!!!!!############
# Path variables
data_path = s3_path + "/BI_IPF_2016/01_data/"
s3_outpath = s3_path + "/BI_IPF_2016/02_result/"
master_path = "/home/hjin/BI_IPF_2016/03_result/"
master_data_path = "/home/hjin/BI_IPF_2016/01_data/"

# data file
pos_file = "all_features_pos.csv"
neg_file ="all_features_neg.csv"
ss_file = 'all_features_score.csv'
exc_file = 'vars_to_exclude.csv'

# Number of outer CV
num_sim = 5

# Number of nested CV
fold = 5

#########Setting End##########################################################

# Don't need to be setup
# seed
random.seed(seed)
seed_seq = [random.randint(10, 100) for i in range(num_sim)]

# S3 output folder
start_time = time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
resultDir_s3 = s3_outpath + app_name + "_" + st + "/"

# Master node output folder
resultDir_master = master_path + app_name + '_' + st + "/"
if not os.path.exists(resultDir_master):
    os.makedirs(resultDir_master)

os.chmod(resultDir_master, 0o777)

# To convert dataframe to LabeledPoint
def parsePoint(line):
    return LabeledPoint(line.label, line.features)

# To top code 4 count varible and drop the original 4 count variable
def topcoding(dataset):
    dataset2 = dataset\
        .withColumn('SYMP_CNT2', when(dataset.SYMP_CNT>7, 7).otherwise(dataset.SYMP_CNT))\
        .withColumn('COMOR_CNT2', when(dataset.COMOR_CNT > 5, 5). otherwise(dataset.COMOR_CNT))\
        .withColumn('NRCOMOR_CNT2', when(dataset.NRCOMOR_CNT>11, 11).otherwise(dataset.NRCOMOR_CNT))\
        .withColumn('PROC_CNT2', when(dataset.PROC_CNT>5, 5).otherwise(dataset.PROC_CNT))\
        .drop('SYMP_CNT').drop('COMOR_CNT').drop('NRCOMOR_CNT')\
        .drop('PROC_CNT')
    return dataset2

# To change patid to matched_positive_id and nonipf_patid to patid
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

# To register dataframe as table
def regit(table, isim, i,name):
    return sqlContext.registerDataFrameAsTable(table[isim][i], (name + str(isim)))

# To create SQL query to union predicted score
def sqlqu1(nsim,name):
    iquery = 'SELECT * FROM ' + name + '0 UNION ALL '
    for i in range(1, nsim-1):
        iquery = iquery + 'SELECT * FROM ' + name +  str(i) + ' UNION ALL '
    query = iquery + 'SELECT * FROM ' + name +  str(nsim-1)
    return query

# To create SQL query to calcualte average probability
def sqlqu2(nIter, name):
    sqla = 'SELECT tb0.patid AS patid, tb0.label AS label, (tb0.prob_1'
    for i in range(1, nIter):
        sqla = sqla + '+tb' + str(i) + '.prob_1'
    sqlb = ')/' + str(nIter) + ' AS avg_prob FROM ' + name + '0 AS tb0'
    for j in range(1, nIter):
        sqlb = sqlb + ' INNER JOIN ' + name + str(j) + ' AS tb' + str(j)\
               + ' ON tb0.patid = tb' + str(j) +'.patid'
    sqlquery = sqla + sqlb
    return sqlquery

# To use the SQL query to create dataframe
def integ_re(nested_re, num, name, iloc, type):
    if type == 'union':
        #register each result into temp table
        for d in range(num):
            regit(nested_re, d, iloc, name)
        #create the query statment
        sqlquery = sqlqu1(num, name)
    elif type == 'avg':
        #register each result into temp table
        for d in range(num):
            regit(nested_re, d,  iloc, name)
        #create the query statment
        sqlquery = sqlqu2(num, name)
    #combine results from simulations
    pred_re = sqlContext.sql(sqlquery)
    return pred_re


# To read in the excluding variable names and generate excluded variable list
def exc_list(d_path, file):
    data = np.loadtxt(d_path + file ,dtype=np.str,delimiter=',',skiprows=0)
    var_ls = data[1:, 0].tolist()
    var_flag_ls = [x + '_FLAG' for x in var_ls]
    var_avg_ls = [x + '_AVG_RXDX' for x in var_ls]
    var_exc_ls = var_flag_ls + var_avg_ls
    return var_exc_ls

# To assemble multiple columns into features
def assembler(dataset, inputcol, outputcol):
    #combine features
    assembler = VectorAssembler(inputCols=inputcol,outputCol=outputcol)
    dataset_asmbl = assembler.transform(dataset)\
        .select('matched_positive_id', 'label', 'patid', outputcol)
    dataset_ori = dataset_asmbl.withColumn('label', dataset_asmbl['label'].cast('double'))
    return dataset_ori

# To subset the data with whose predicted score > pos_25_score,
# To calculate new variable logprob_1 = log10(prob_1)
# To assemble logprob_1 into features
def subdata(preddata, dataset, pos_25_score, flag ):
    if flag == 'tr':
        sub_id = preddata.filter(preddata.prob_1 >= pos_25_score)\
            .withColumn('logprob_1', log(10.0, preddata.prob_1))\
            .select('patid', 'logprob_1')
    elif flag == 'avg':
        sub_id = preddata.filter(preddata.avg_prob >= pos_25_score)\
            .withColumn('logprob_1', log(10.0, preddata.avg_prob))\
            .select('patid', 'logprob_1')
    dataset = dataset.withColumnRenamed('features', 'features1')
    sub_dataset = dataset.join(sub_id, dataset.patid == sub_id.patid, 'inner')\
        .select(dataset.matched_positive_id, dataset.patid,
                dataset.label, dataset.features1, sub_id.logprob_1)
    sub_dataset2 = assembler(sub_dataset, ['features1', 'logprob_1'],'features')
    return sub_dataset2

# nested 5 fold CV
def nested_func(ifold, trfoldid, trsim, tssim, repsim):

    #get the cv training and test data
    cvtsid = trfoldid.filter(trfoldid.foldid == ifold)
    cvts = trsim\
        .join(cvtsid, trsim.matched_positive_id==cvtsid.matched_positive_id, 'inner')\
        .select(trsim.matched_positive_id, trsim.label, trsim.patid, trsim.features)
    cvtr = trsim.subtract(cvts)
    cvtrrdd = cvtr.map(parsePoint)
    cv_model = LogisticRegressionWithLBFGS.train(cvtrrdd, intercept=True,regType=None)

    #clear the threshold
    cv_model.clearThreshold()

    #predict on test data
    scoreAndLabels_cvtr = cvtr.map(lambda p:
                               (float(cv_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_cvts = cvts.map(lambda p:
                               (float(cv_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_tssim = tssim.map(lambda p:
                               (float(cv_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_repsim = repsim.map(lambda p:
                               (float(cv_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))

    pred_score_cvtr = scoreAndLabels_cvtr.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')
    pred_score_cvts = scoreAndLabels_cvts.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')
    pred_score_tssim = scoreAndLabels_tssim.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')
    pred_score_repsim = scoreAndLabels_repsim.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')
    pred_score_reptssim = pred_score_tssim.unionAll(pred_score_repsim)

    #predict on dataset
    cvtr_scoreLabel = pred_score_cvtr.select('prob_1', 'label').rdd
    cvts_scoreLabel = pred_score_cvts.select('prob_1', 'label').rdd

    #AUC & AUPR
    AUPR_cvtr = BinaryClassificationMetrics(cvtr_scoreLabel).areaUnderPR
    AUPR_cvts = BinaryClassificationMetrics(cvts_scoreLabel).areaUnderPR

    return [pred_score_cvts, pred_score_tssim, pred_score_reptssim, AUPR_cvtr,
            AUPR_cvts]


# Outer 5-fold CV, in each fold run 2nd stage model
def sim_function(isim, patsim, dataset, ss_ori, fold):

    #1st stage model
    #select patients in each simulation from patsim
    tssimid = patsim.filter(patsim.simid == isim)
    trsimid = patsim.subtract(tssimid).drop(patsim.simid)

    #create fold ID for nested 5-fold CV
    trfoldid = addID(trsimid, fold, par, 'foldid')

    #select corresponding trainning and test set and rep sample
    tssim = dataset\
        .join(tssimid, tssimid.matched_positive_id==dataset.matched_positive_id,
              'inner')\
        .select(dataset.matched_positive_id, dataset.label, dataset.patid,
                dataset.features)

    trsim = dataset.subtract(tssim)

    repsim = ss_ori\
        .join(tssimid, tssimid.matched_positive_id==ss_ori.matched_positive_id,
              'inner')\
        .select(ss_ori.matched_positive_id, ss_ori.label, ss_ori.patid,
                ss_ori.features)

    trfoldid.cache()
    trsim.cache()
    tssim.cache()
    repsim.cache()

    #run nested 5-fold CV
    start = datetime.datetime.now().replace(microsecond=0)
    nested_re = [nested_func(ifold, trfoldid=trfoldid, trsim=trsim,
                             tssim=tssim, repsim=repsim) for ifold in range(fold)]
    end = datetime.datetime.now().replace(microsecond=0)
    print('parallel running time is ', end - start)
    trfoldid.unpersist()

    #combine the predicted score from each fold
    #for training it's union, for test and rep sample it's average
    comb_tr = integ_re(nested_re=nested_re, num=fold, name='comb_tr', iloc=0,
                       type='union')
    avg_ts = integ_re(nested_re=nested_re, num=fold, name='avg_ts', iloc=1,
                       type='avg')
    avg_repts = integ_re(nested_re=nested_re, num=fold, name='avg_repts',
                         iloc=2, type='avg')

    #output the predicted score
    comb_tr.save((resultDir_s3+'pred_score_1stcombtr_sim'+str(isim)),
                                 "com.databricks.spark.csv",header="true")

    avg_ts.save((resultDir_s3+'pred_score_1stavgts_sim'+str(isim)),
                                 "com.databricks.spark.csv",header="true")

    avg_repts.save((resultDir_s3+'pred_score_1stavgrepts_sim'+str(isim)),
                                      "com.databricks.spark.csv",header="true")

    #calculate the mean of cv AUPR
    avg_aupr_cvtr = np.mean([x[3] for x in nested_re])
    avg_aupr_cvts = np.mean([x[4] for x in nested_re])

    #print out AUPR results
    avg_aupr = "CV Training data AUPR = %s " % avg_aupr_cvtr + "\n" \
           +  "CV Test data AUPR = %s " % avg_aupr_cvts
    auc_file = open(resultDir_master + '1stavgAUPR_sim' + str(isim) +'.txt',"w")
    auc_file.writelines(avg_aupr)
    auc_file.close()
    os.chmod(resultDir_master + '1stavgAUPR_sim' + str(isim) +'.txt', 0o777)

    #calculate pos_25_score
    pred_pos = comb_tr.filter(comb_tr.label == 1).select(comb_tr.prob_1).collect()
    pred_pos_array = np.array(pred_pos)
    pos_25_score = np.percentile(pred_pos_array, 75)

    #2nd stage model
    #sub training data
    sub_trsim = subdata(comb_tr, trsim, pos_25_score, 'tr')
    sub_tssim = subdata(avg_ts, tssim, pos_25_score, 'avg')
    sub_repsim = subdata(avg_repts, repsim, pos_25_score, 'avg')

    sub_trsim.cache()
    sub_tssim.cache()
    sub_repsim.cache()

    #get LabeledandPoint rdd data
    sub_trsimrdd = sub_trsim.map(parsePoint)

    # Build the model
    sim_model = LogisticRegressionWithLBFGS.train(sub_trsimrdd, intercept=True,regType=None)

    #clear the threshold
    sim_model.clearThreshold()

    #predict on test data
    scoreAndLabels_subtr = sub_trsim.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_subts = sub_tssim.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_subrep = sub_repsim.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))

    #Identify the probility of response
    pred_score_subtr = scoreAndLabels_subtr.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')

    pred_score_subts = scoreAndLabels_subts.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')

    pred_score_subrep = scoreAndLabels_subrep.toDF()\
        .withColumnRenamed('_1', 'prob_1')\
        .withColumnRenamed('_2', 'label')\
        .withColumnRenamed('_3', 'patid')\
        .withColumnRenamed('_4', 'matched_positive_id')

    pred_score_subrepts = pred_score_subts.unionAll(pred_score_subrep)

    #get the scoreLabel RDD
    subtr_scoreLabel = pred_score_subtr.select('prob_1', 'label').rdd
    subts_scoreLabel = pred_score_subts.select('prob_1', 'label').rdd

    #AUPR
    AUPR_subtr = BinaryClassificationMetrics(subtr_scoreLabel).areaUnderPR
    AUPR_subts = BinaryClassificationMetrics(subts_scoreLabel).areaUnderPR

    #print out AUPR results
    sub_aupr = "sub Training data AUPR = %s " % AUPR_subtr + "\n" \
           +  "sub Test data AUPR = %s " % AUPR_subts
    auc_file = open(resultDir_master + '2ndsubAUPR_sim' + str(isim) +'.txt',"w")
    auc_file.writelines(sub_aupr)
    auc_file.close()
    os.chmod(resultDir_master + '2ndsubAUPR_sim' + str(isim) +'.txt', 0o777)

    pred_score_subtr.save((resultDir_s3+'pred_score_2ndsubtr_sim'+str(isim)),
                                 "com.databricks.spark.csv",header="true")

    pred_score_subts.save((resultDir_s3+'pred_score_2ndsubts_sim'+str(isim)),
                                 "com.databricks.spark.csv",header="true")

    pred_score_subrepts.save((resultDir_s3+'pred_score_2ndsubrepts_sim'+str(isim)),
                                      "com.databricks.spark.csv",header="true")

    trsim.unpersist()
    tssim.unpersist()
    repsim.unpersist()
    sub_trsim.unpersist()
    sub_tssim.unpersist()
    sub_repsim.unpersist()

    return [pred_score_subts, pred_score_subrepts]

#define the main function
def main(sc, data_path=data_path, pos_file=pos_file, ss_file=ss_file,
         neg_file= neg_file, num_sim=num_sim, seed=seed,
         seed_seq=seed_seq, par=par, resultDir_s3=resultDir_s3,
         resultDir_master=resultDir_master, fold=fold):

    #reading in the data from S3
    pos2 = sqlContext.read.load((data_path + pos_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

    neg2 = sqlContext.read.load((data_path + neg_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')
    ss2 = sqlContext.read.load((data_path + ss_file),
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

    #top coding 4 count variables
    pos = topcoding(pos2)
    neg = topcoding(neg2)
    ss = topcoding(ss2)

    #rename patid etc..
    pos = drop_rename(pos, pos_file)
    neg = drop_rename(neg, neg_file)
    ss = drop_rename(ss, ss_file)

    #reading in excluded variable list from master node
    exc_var_list = exc_list(master_data_path, exc_file)

    #see the column names
    pos_col = pos.columns

    #include variable list
    common_list = ['matched_positive_id', 'label', 'patid']
    #orla would like to remove "LVL3_GASTROSC_AVG_RXDX" as well on 0824
    inc_var = [x for x in pos_col if x not in
               exc_var_list+common_list+['LVL3_GASTROSC_AVG_RXDX']]

    #combine features
    pos_ori = assembler(pos, inc_var, 'features')
    neg_ori = assembler(neg, inc_var, 'features')
    ss_ori = assembler(ss, inc_var, 'features')

    #union All positive and negative data as dataset
    dataset = pos_ori.unionAll(neg_ori)

    #create a dataframe which has 2 column, 1 is patient ID, other one is simid
    patid_pos = pos_ori.select('matched_positive_id')
    patsim = addID(patid_pos, num_sim, par, 'simid')

    patsim.cache()
    sim_result_ls = [sim_function(isim, patsim=patsim, dataset=dataset,
                                  ss_ori=ss_ori, fold=fold)
                 for isim in range(num_sim)]

    patsim.unpersist()

    #comb_subts = integ_re(nested_re=sim_result_ls, num=num_sim,
    #                     name='comb_subts', iloc=0,type='union')
    #comb_subrepts = integ_re(nested_re=sim_result_ls, num=num_sim,
    #                      name='comb_subrepts', iloc=1,type='union')

    #output the predicted score on test data
    #comb_subts.save((resultDir_s3+'pred_score_2ndsubts'),
    #                             "com.databricks.spark.csv",header="true")
    #comb_subrepts.save((resultDir_s3+'pred_score_2ndsubrepts'),
    #                             "com.databricks.spark.csv",header="true")

if __name__ == "__main__":
    
    sc = SparkContext(appName = app_name)
    sqlContext = SQLContext(sc)

    #call main function
    main(sc)

    sc.stop()
    




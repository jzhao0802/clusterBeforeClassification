from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os
import time
import datetime
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from imspaeva import BinaryClassificationEvaluatorWithPrecisionAtRecall

# from stratification.py import AppendDataMatchingFoldIDs

# def dimension_reduction(org_data):
    

# def record_experiment_info(result_dir_master):
    # pass

# To register dataframe as table
def regit(table, isim, i,name):
    the following is problematic
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
    
# Outer 5-fold CV, in each fold run 2nd stage model
def sim_function(isim, patsim, dataset, ss_ori, fold, par):

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

    #calculate the mean of cv AUPR
    avg_aupr_cvtr = np.mean([x[3] for x in nested_re])
    avg_aupr_cvts = np.mean([x[4] for x in nested_re])

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
    scoreAndLabels_subtr = sub_trsim.rdd.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_subts = sub_tssim.rdd.map(lambda p:
                               (float(sim_model.predict(p.features)),p.label,
                                p.patid, p.matched_positive_id))
    scoreAndLabels_subrep = sub_repsim.rdd.map(lambda p:
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

    pred_score_subrepts = pred_score_subts.union(pred_score_subrep)

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

    return pred_score_subrepts

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
    
def cross_evaluate(data_pn, data_ss, n_eval_folds, n_inner_folds, par):
    
    patid_pos = data_pn.select('matched_positive_id').distinct()
    patsim = addID(patid_pos, n_eval_folds, par, 'simid')

    patsim.cache()
    sim_result_ls = [sim_function(isim, patsim=patsim, dataset=data_pn,
                                  ss_ori=data_ss, fold=n_inner_folds, par=par)
                 for isim in range(num_sim)]

    patsim.unpersist()
    
    return sim_result_ls
    
# To assemble multiple columns into features
def assembler(dataset, inputcol, outputcol):
    #combine features
    assembler = VectorAssembler(inputCols=inputcol,outputCol=outputcol)
    dataset_asmbl = assembler.transform(dataset)\
        .select('matched_positive_id', 'label', 'patid', outputcol)
    dataset_ori = dataset_asmbl.withColumn('label', dataset_asmbl['label'].cast('double'))
    return dataset_ori
    
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

    # union All positive and negative data as dataset
    dataset_pn = pos_assembled.union(neg_assembled)
    
    # 
    # 'prob_1' and 'label'
    eval_result = cross_evaluate(dataset_pn, ss_assembled, labelCol, featureCol, matchedPosIDCol, n_eval_folds, par)
    eval_result_unioned = eval_result[0]
    for i in range(1, len(eval_result_unioned)):
        eval_result_unioned = eval_result_unioned.union(eval_result[i])
    
    evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="raw")
    desired_recalls = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
    metricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": x}} for x in desired_recalls]
    all_metrics = evaluator.evaluateWithSeveralMetrics(dataset, metricSets = metricSets)
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/" + st + "/"
    all_metrics.coalesce(1).write.csv(result_dir_s3 + "prs_rep.csv", header="true")
    

if __name__ == "__main__":
    main_data_path = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/Results/20161207_101137/"
    main_pos_file = "pos.csv"
    main_neg_file = "neg.csv"
    main_ss_file = "ss.csv"
    main_num_sim = 3
    main(main_data_path, main_pos_file, main_neg_file, main_ss_file, main_num_sim)
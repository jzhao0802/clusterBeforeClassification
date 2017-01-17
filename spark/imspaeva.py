from abc import abstractmethod, ABCMeta
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import keyword_only
from pyspark.mllib.common import inherit_doc
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasRawPredictionCol
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession

__all__ = ['BinaryClassificationEvaluatorWithPrecisionAtRecall']

partition_size = 20

def getitem(i):
    def getitem_(v):
         return v.array.item(i)
    return F.udf(getitem_, DoubleType())


def precision_recall_curve(labelAndVectorisedScores, rawPredictionCol, labelCol):
    # get the tps, fps and thresholds
    tpsFpsScorethresholds = _binary_clf_curve(labelAndVectorisedScores,
                                              rawPredictionCol, labelCol)
    # total tps
    tpsMax = ((tpsFpsScorethresholds.agg(F.max(F.col("tps"))).collect())[0].asDict())["max(tps)"]
    
    # calculate precision
    tpsFpsScorethresholds = tpsFpsScorethresholds \
        .withColumn("precision", F.col("tps") / (F.col("tps") + F.col("fps") + 1e-12))
    
    # calculate recall
    dummy_pr = SparkSession.builder.getOrCreate().createDataFrame([(1,0)],["precision","recall"])
    tpsFpsScorethresholds = dummy_pr.union(tpsFpsScorethresholds \
                                          .withColumn("recall", F.col("tps") / tpsMax)\
                                          .select("precision", "recall"))
    
    return tpsFpsScorethresholds


def _binary_clf_curve(labelAndVectorisedScores, rawPredictionCol, labelCol):
    
    # sort the dataframe by pred column in descending order
    localPosProbCol = "pos_probability"
    labelAndPositiveProb = labelAndVectorisedScores.select(labelCol, getitem(1)(rawPredictionCol).alias(localPosProbCol))
    
    # round the fractional prediction column
    labelAndPositiveProb = labelAndPositiveProb\
        .withColumn("_tmp_pred", F.round(localPosProbCol, 3))\
        .drop(localPosProbCol)\
        .withColumnRenamed("_tmp_pred", localPosProbCol)\
        .sort(F.desc(localPosProbCol))
    
    # adding index to the dataframe
    sortedScoresAndLabels = labelAndPositiveProb.rdd.zipWithIndex() \
        .toDF(['data', 'index']) \
        .select('data.' + labelCol, 'data.' + localPosProbCol, "index")
    
    groupSumLabelCol = "group_sum_labels"
    groupMaxIndexCol = "group_max_indices"
    sortedScoresAndLabels = sortedScoresAndLabels\
        .groupBy([localPosProbCol, labelCol])\
        .agg(F.sum(labelCol).alias(groupSumLabelCol), F.max("index").alias(groupMaxIndexCol))
    
    # sortedScoresAndLabels = labelAndPositiveProb.sort(F.desc(localPosProbCol))
    
    # creating rank for pred column
    lookup = (sortedScoresAndLabels.select(localPosProbCol)
              .distinct()
              .sort(F.desc(localPosProbCol))
              .rdd
              .zipWithIndex()
              .map(lambda x: x[0] + (x[1],))
              .toDF([localPosProbCol, "rank"]))
    
    # join the dataframe with lookup to assign the ranks
    sortedScoresAndLabels = sortedScoresAndLabels.join(lookup, [localPosProbCol])
    
    # sorting in descending order based on the pred column
    sortedScoresAndLabels = sortedScoresAndLabels.sort(groupMaxIndexCol)
    
    
    
    # saving the dataframe to temporary table
    sortedScoresAndLabels.registerTempTable("processeddata")
    
    # TODO: script to avoid partition by warning, and span data across clusters nodes
    # creating the cumulative sum for tps
    sortedScoresAndLabelsCumSum = labelAndVectorisedScores.sql_ctx \
        .sql(
        "SELECT " + labelCol + ", " + localPosProbCol + ", " + groupSumLabelCol + ", rank, " + groupMaxIndexCol + ", sum(" + groupSumLabelCol + ") OVER (ORDER BY " + groupMaxIndexCol + ") as tps FROM processeddata ")
    
    # repartitioning
    sortedScoresAndLabelsCumSum = sortedScoresAndLabelsCumSum.coalesce(partition_size)
    
    # # cache after partitioning
    sortedScoresAndLabelsCumSum.cache()
    
    # retain only the group-wise (according to threshold) max tps
    
    df_max_tps_in_group = sortedScoresAndLabelsCumSum.groupBy(localPosProbCol).agg(F.max("tps").alias("max_tps"))
    dup_removed_scores_labels = \
        sortedScoresAndLabelsCumSum.join(
            df_max_tps_in_group,
            [sortedScoresAndLabelsCumSum[localPosProbCol] == df_max_tps_in_group[localPosProbCol],
             sortedScoresAndLabelsCumSum["tps"] == df_max_tps_in_group["max_tps"]],
            how="right_outer"
        )\
        .drop(df_max_tps_in_group[localPosProbCol])\
        .drop(df_max_tps_in_group["max_tps"])\
        .groupBy([localPosProbCol, "tps"])\
        .agg(F.max(groupMaxIndexCol).alias("max_index"))
    
    # creating the fps column based on rank and tps column
    df_with_fps = dup_removed_scores_labels \
        .withColumn("fps", 1 + F.col("max_index") - F.col("tps"))
        
    return df_with_fps

def getPrecisionAtOneRecallFromPRCurve(curve, recall):
    pr_curve_with_recall_diff = curve\
        .withColumn("recall_diff", F.abs(F.col("recall") - recall))
    min_recall_diff = pr_curve_with_recall_diff\
        .agg(F.min("recall_diff")\
        .alias("min_recall_diff"))\
        .collect()[0].asDict()["min_recall_diff"]
    precision = pr_curve_with_recall_diff\
        .filter(F.abs(F.col("recall_diff") - min_recall_diff) < 1e-9)\
        .sort("recall", F.desc("precision"))\
        .first().asDict()["precision"]
        
    return precision 
    
    
# 1. The precision is from a pair whose recall is the closest to the desired recall. 
# 2. If there are multiple pairs the same close to the desired recall, choose the pair
#    with the smallest recall (thus the highest threshold).
# 3. If there are multiple pairs satisfying 1 and 2, choose the largest precision 
#    (again, this corresponds to the highest threshold)
def getPrecisionAtOneRecall(labelAndVectorisedScores,
                         rawPredictionCol,
                         labelCol,
                         desired_recall):                         
    # get precision, recall, thresholds
    prcurve = precision_recall_curve(labelAndVectorisedScores, rawPredictionCol, labelCol)
    precision = getPrecisionAtOneRecallFromPRCurve(prcurve, desired_recall)
        
    return precision 

def getPrecisionAtMultipleRecalls(labelAndVectorisedScores,
                         rawPredictionCol,
                         labelCol,
                         desired_recalls):
    prcurve = precision_recall_curve(labelAndVectorisedScores, rawPredictionCol, labelCol)
    return map(lambda x: getPrecisionAtOneRecallFromPRCurve(prcurve, x), desired_recalls)
    
@inherit_doc
class BinaryClassificationEvaluatorWithPrecisionAtRecall(BinaryClassificationEvaluator, HasLabelCol, HasRawPredictionCol):
    """
    
    Evaluator for binary classification, which expects two input
    columns: rawPrediction and label.

    >>> from pyspark.ml.linalg import Vectors
    >>> scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
    ...    [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
    >>> dataset = sqlContext.createDataFrame(scoreAndLabels, ["raw", "label"])
    ...
    >>> evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(
    ...    rawPredictionCol="raw",
    ...     metricName="precisionAtGivenRecall",
    ...     metricParams={"recallValue":1}
    ... )
    >>> evaluator.evaluate(dataset)
    0.57...
    >>> evaluator.evaluate(dataset, {evaluator.metricName: "areaUnderPR"})
    0.70...
    
    """
    
    # a placeholder to make it appear in the generated doc
    metricName = Param(Params._dummy(), "metricName",
                       "metric name in evaluation (areaUnderROC|areaUnderPR)")
    
    
    @keyword_only
    def __init__(self, rawPredictionCol="rawPrediction", labelCol="label",
                 metricName="areaUnderROC", metricParams={"recallValue": 0.6}):
        """
        __init__(self, rawPredictionCol="rawPrediction", labelCol="label", \
                 metricName="areaUnderROC", metricParams={"recallValue": 0.6})
        Currently 'metricParams' is only used when 'metricName' is "precisionAtGivenRecall"
        """
        super(BinaryClassificationEvaluatorWithPrecisionAtRecall.__mro__[1], self).__init__()
        if (metricName == "areaUnderROC") | (metricName == "areaUnderPR"):
            self._java_obj = self._new_java_obj(
                "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator", self.uid)
            #: param for metric name in evaluation (areaUnderROC|areaUnderPR)
            self.metricName = Param(self, "metricName",
                                    "metric name in evaluation (areaUnderROC|areaUnderPR)")
            self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                             metricName="areaUnderROC")
            kwargs = self.__init__._input_kwargs
            if "metricParams" in kwargs.keys():
                kwargs.pop("metricParams")
            
        elif (metricName == "precisionAtGivenRecall"):
            self.metricParams = Param(
                self, "metricParams", "additional parameters for calculating the metric, such as the recall value in getPrecisionAtOneRecall")
            self.metricName = Param(self, "metricName",
                                    "metric name in evaluation (areaUnderROC|areaUnderPR)")
            self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                             metricName="areaUnderROC", metricParams={"recallValue": 0.6})
            kwargs = self.__init__._input_kwargs
            
        else:
            raise ValueError("Invalid input metricName: {}".format(self.metricName))
            
        self._set(**kwargs)
        
        # for the computing precision at given recall in PySpark (in case it's only requested in calling evaluate())
        self.initMetricParams = metricParams
        self.initMetricNameValue = metricName
        self.rawPredictionColValue = rawPredictionCol
        self.labelColValue = labelCol
        
    def _cal_pr_curve(self, labelAndVectorisedScores):
        """
        Calculate the precision-recall (PR) curve. It's not a public method. 
        Use it only when you understand what you are doing exactly. 
        
        The PR curve result is similar to that in sklearn and ROCR in R with minor difference. The first 
        precision-recall pair always takes the value (1,0). 
        
        """
        rawPredictionCol = self.rawPredictionColValue
        labelCol = self.labelColValue
        curve = precision_recall_curve(labelAndVectorisedScores, rawPredictionCol, labelCol).select("precision","recall")
        
        return curve
    
    def isLargerBetter(self):
        return True
    
    def evaluate(self, dataset, params=None):
        """
        evaluate(self, dataset, params=None)
        
        Input:
        dataset: a DataFrame containing a label column and a prediction column. Every element in the prediction column
                 must be a Vector containing the predicted scores to the negative (first element) and positive (second
                 element) classes.
        params: parameters about the metric to use. This will overwrite the settings in __init__().
        Output: the evaluated metric value.
        """
        if params is None:
            if (self.initMetricNameValue == "areaUnderROC") | (self.initMetricNameValue == "areaUnderPR"):
                return super(BinaryClassificationEvaluatorWithPrecisionAtRecall.__mro__[1], self).evaluate(dataset)
            else:
                if "recallValue" in self.initMetricParams.keys():
                    return getPrecisionAtOneRecall(dataset,
                                                self.rawPredictionColValue,
                                                self.labelColValue,
                                                self.initMetricParams["recallValue"])
                else:
                    raise ValueError("To compute 'precisionAtGivenRecall', metricParams must include the key 'recallValue'.")
        elif (isinstance(params, dict)):
            if "precisionAtGivenRecall" in params.values():
                if "metricParams" in params.keys():
                    if "recallValue" in params["metricParams"].keys():
                        return getPrecisionAtOneRecall(dataset,
                                                    self.rawPredictionColValue,
                                                    self.labelColValue,
                                                    params["metricParams"]["recallValue"])
                    else:
                        raise ValueError("To compute 'precisionAtGivenRecall', metricParams must include the key 'recallValue'.")
                else:
                    raise ValueError("When 'precisionAtGivenRecall' is specified calling the evaluate() method, " + \
                                     "'metricParams' must also be specified.")
            elif "precisionAtGivenMultipleRecalls" in params.values():
                if "metricParams" in params.keys():
                    if "recallValues" in params["metricParams"].keys():
                        return getPrecisionAtMultipleRecalls(dataset,
                                                    self.rawPredictionColValue,
                                                    self.labelColValue,
                                                    params["metricParams"]["recallValues"])
                    else:
                        raise ValueError("To compute 'precisionAtGivenMultipleRecalls', metricParams must include the key 'recallValues'.")
                else:
                    raise ValueError("When 'precisionAtGivenMultipleRecalls' is specified calling the evaluate() method, " + \
                                     "'metricParams' must also be specified.")
            else:
                return super(BinaryClassificationEvaluatorWithPrecisionAtRecall.__mro__[1], self).evaluate(dataset, params)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))
    
    def evaluateWithSeveralMetrics(self, dataset, metricSets=None):
        """
        Evaluate the performance using a number of metrics, including areaUnderROC, areaUnderPR 
        and precisionAtGivenRecall at different recall values. 
        
        Input:
        dataset: a DataFrame containing a label column and a prediction column. Every element in the prediction column
                 must be a Vector containing the predicted scores to the negative (first element) and positive (second
                 element) classes.
        metricSets: a list / tuple of metrics to use. 
        Output: A list of metric values, each corresponding to a desired metric. 
        """
        if metricSets is None: # all metrics
            metricSets = [{"metricName": "areaUnderROC"},
                          {"metricName": "areaUnderPR"},
                          {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.05}}]            
        resultMetricSets = [None for _ in range(len(metricSets))]
        pagrs = []
        for i in range(len(metricSets)):
            params = metricSets[i]
            if params["metricName"] != "precisionAtGivenRecall":
                value = self.evaluate(dataset, params)
                if len(params.keys()) == 1:
                    key = params["metricName"]
                else:
                    key = params["metricName"] + " at recallValue " + str(params["metricParams"]["recallValue"])
                resultMetricSets[i] = {key:value}
            else: 
                pagrs.append([i,params["metricParams"]["recallValue"]])
                continue
        if None in resultMetricSets:
            pr_params = {"metricName": "precisionAtGivenMultipleRecalls", "metricParams": {"recallValues": [x[1] for x in pagrs]}}
            precisions = self.evaluate(dataset, pr_params)
            i = 0
            for item in pagrs:
                key = "precisionAtGivenRecall" + " at recallValue " + str(pagrs[i][1])
                resultMetricSets[item[0]] = {key:precisions[i]}
                i += 1          
            
        return resultMetricSets
    
    def setMetricName(self, value):
        """
        Sets the value of :py:attr:`metricName`.
        """
        self._paramMap[self.metricName] = value
        return self
    
    def getMetricName(self):
        """
        Gets the value of metricName or its default value.
        """
        return self.getOrDefault(self.metricName)
    
    def setMetricParams(self, value):
        """
        Sets the value of :py:attr:`metricParams`.
        """
        self._paramMap[self.metricParams] = value
        return self
    
    def getMetricValue(self):
        """
        Gets the value of metricParams or its default value.
        """
        return self.getOrDefault(self.metricParams)
    
    @keyword_only
    def setParams(self, rawPredictionCol="rawPrediction", labelCol="label",
                  metricName="areaUnderROC", metricParams={"recallValue": 0.6}):
        """
        setParams(self, rawPredictionCol="rawPrediction", labelCol="label", \
                  metricName="areaUnderROC")
        Sets params for binary classification evaluator.
        """
        tmp = getattr(self, "metricParams", None)
        if tmp is None:
            self.metricParams = Param(self, "metricParams",
                "additional parameters for calculating the metric, such as the recall value in getPrecisionAtOneRecall")
        kwargs = self.setParams._input_kwargs
        self.initMetricParams = metricParams
        self.initMetricNameValue = metricName
        self.rawPredictionColValue = rawPredictionCol
        self.labelColValue = labelCol
        return self._set(**kwargs)
        
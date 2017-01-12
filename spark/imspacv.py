from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.sql.functions import lit
from pyspark import since, keyword_only
from pyspark.sql.types import IntegerType
import numpy
import sys
from imspaeva import BinaryClassificationEvaluatorWithPrecisionAtRecall

__all__ = ["CrossValidatorWithStratificationID"]


def _write_to_np_MetricValueEveryParamSet(listMetricValueEveryParamSet, 
                                          colnamesMetricValueEveryParamSet, 
                                          j, params, metricForCV, otherMetricResults=None):
    if otherMetricResults is None:
        nOtherMetrics = 0
    else:
        nOtherMetrics = len(otherMetricResults)
    
    #
    for paramNameStruct in params.keys():
        paramName = paramNameStruct.name
        paramVal = params[paramNameStruct]
        try:
            colID = colnamesMetricValueEveryParamSet.index(paramName)
        except ValueError:
            sys.stderr.write("Error! "\
            + paramName + \
            " doesn't exist in the list of hyper-parameter names colnamesMetricValueEveryParamSet.\n")
        
        if type(paramVal) is str:
            listMetricValueEveryParamSet[j][colID] = paramVal
        else:
            listMetricValueEveryParamSet[j][colID] = float(paramVal)
        
    listMetricValueEveryParamSet[j][-1-nOtherMetrics] += float(metricForCV)
    
    if nOtherMetrics > 0:
        for iMetricResult in range(nOtherMetrics):
            tupleResult = list(otherMetricResults[iMetricResult].items())[0]
            metricName = tupleResult[0]
            metricValue = tupleResult[1]
            try:
                colID = colnamesMetricValueEveryParamSet.index(metricName)
            except ValueError:
                sys.stderr.write("Error! "\
                + metricName + \
                " doesn't exist in the list of other metric names colnamesMetricValueEveryParamSet.\n")
            listMetricValueEveryParamSet[j][iMetricResult-nOtherMetrics] += float(metricValue)        
    
    return listMetricValueEveryParamSet
    

class CrossValidatorWithStratificationID(CrossValidator):
    """
    Similar to the built-in pyspark.ml.tuning.CrossValidator, but the stratification will be done
    according to the fold IDs in a column of the input data. 
    
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.evaluation import BinaryClassificationEvaluator
    >>> from pyspark.ml.linalg import Vectors
    >>> dataset = spark.createDataFrame(
    ...     [(Vectors.dense([0.0]), 0.0, 0),
    ...      (Vectors.dense([0.4]), 1.0, 1),
    ...      (Vectors.dense([0.5]), 0.0, 2),
    ...      (Vectors.dense([0.6]), 1.0, 3),
    ...      (Vectors.dense([1.0]), 1.0, 4)] * 10,
    ...     ["features", "label", "foldID"])
    >>> lr = LogisticRegression()
    >>> grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    >>> evaluator = BinaryClassificationEvaluator()
    >>> cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, stratifyCol="foldID")
    >>> cvModel = cv.fit(dataset)
    >>> cvModel.avgMetrics.show()
    +----------+-------+-----------+
    |paramSetID|maxIter|metricValue|
    +----------+-------+-----------+
    |         0|    0.0|        0.6|
    |         1|    1.0|        0.6|
    +----------+-------+-----------+
    >>> evaluator.evaluate(cvModel.transform(dataset))
    0.5
    
    """
    
    # a placeholder to make it appear in the generated doc
    stratifyCol = Param(Params._dummy(), "stratifyCol", "column name of the stratification ID")
    evaluateOtherMetrics = Param(Params._dummy(), "evaluateOtherMetrics", "Boolean flag whether to evaluate with other metrics")
    otherMetrics = Param(Params._dummy(), "otherMetrics", "dictionary specifying other metrics to evaluate")
    
    
    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None, 
                 evaluateOtherMetrics=False, otherMetrics=None):
        """
        __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None,
                 evaluateOtherMetrics=False, otherMetrics=None)
        """     
        if stratifyCol is None:
            raise ValueError("stratifyCol must be specified.")        
        super(CrossValidatorWithStratificationID, self).__init__()
        self.bestIndex = None
        if evaluateOtherMetrics:
            if not isinstance(evaluator, BinaryClassificationEvaluatorWithPrecisionAtRecall):
                raise TypeError("When evaluateOtherMetrics is set True, the evaluator must be of class BinaryClassificationEvaluatorWithPrecisionAtRecall.")        
        self._setDefault(evaluateOtherMetrics=False, otherMetrics=None)
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)
        
    @keyword_only
    @since("1.4.0")
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None,
                  evaluateOtherMetrics=False, otherMetrics=None):
        """
        setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, stratifyCol=None,
                  evaluateOtherMetrics=False, otherMetrics=None):
        Sets params for cross validator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)
        
    @keyword_only
    @since("1.4.0")
    def setNumFolds(self, value):
        """
        Deprecated.
        """
        raise AttributeError(type(self).__name__ + " does not have method 'setNumFolds'")
        
    @since("1.4.0")
    def getNumFolds(self):
        """
        Deprecated.
        """
        raise AttributeError(type(self).__name__ + " does not have method 'getNumFolds'")
    
    def _fit(self, dataset):
        try: 
            stratifyCol = self.getOrDefault(self.stratifyCol)
            dfStratifyCol = dataset.select(stratifyCol)
            rowsStratifyCol = dfStratifyCol.distinct().collect()
            foldIDs = [x[stratifyCol] for x in rowsStratifyCol]
            if (set(foldIDs) != set(range(max(foldIDs)+1))):
                raise ValueError("The stratifyCol column does not have zero-based consecutive integers as fold IDs.")            
        except Exception as e:
            print("Something is wrong with the stratifyCol:")
            raise
        
        # 
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        
        evaluateOtherMetrics = self.getOrDefault(self.evaluateOtherMetrics)
        otherMetrics = self.getOrDefault(self.otherMetrics)
        
        nFolds = dfStratifyCol.distinct().count()
        
        # select features, label and foldID in order
        featuresCol = est.getFeaturesCol()
        labelCol = est.getLabelCol()
        dataWithFoldID = dataset.select(featuresCol, labelCol, stratifyCol)
        
        paramNames = [x.name for x in epm[0].keys()]
        metricValueColForCV = "metric for CV " + eva.getMetricName()
        
        if not evaluateOtherMetrics:
            listMetricValueEveryParamSet = [[None for _ in range(2 + len(paramNames))] for _ in range(len(epm))]
            for i in range(len(epm)):
                listMetricValueEveryParamSet[i][0] = i
                listMetricValueEveryParamSet[i][-1] = 0
        else:
            listMetricValueEveryParamSet = None
        
        colnamesMetricValueEveryParamSet = ["paramSetID"] + paramNames + [metricValueColForCV]
        
        nOtherMetrics = 0
        metricForCVArray = numpy.zeros((nFolds, numModels))
        otherMetricResults = None
        for i in range(nFolds):
            condition = (dataWithFoldID[stratifyCol] == i)    
            validation = dataWithFoldID.filter(condition).select(featuresCol, labelCol)
            train = dataWithFoldID.filter(~condition).select(featuresCol, labelCol)            
            
            for j in range(numModels):
                model = est.fit(train, epm[j])
                # TODO: duplicate evaluator to take extra params from input
                transformed_data = model.transform(validation, epm[j])
                metricForCV = eva.evaluate(transformed_data)
                metricForCVArray[i,j] = metricForCV
                
                if evaluateOtherMetrics:
                    if otherMetrics is None:
                        otherMetricResults = eva.evaluateWithSeveralMetrics(transformed_data)
                    else:
                        otherMetricResults = eva.evaluateWithSeveralMetrics(transformed_data, otherMetrics)
                    nOtherMetrics = len(otherMetricResults)
                    
                    # initialise for other metrics
                    if (i == 0) and (j == 0):
                        listMetricValueEveryParamSet = [[None for _ in range(2 + len(paramNames) + nOtherMetrics)] for _ in range(len(epm))]
                        for ii in range(len(epm)):
                            listMetricValueEveryParamSet[ii][0] = ii
                            for jj in range(-1-nOtherMetrics, 0):
                                listMetricValueEveryParamSet[ii][jj] = 0
                        colnamesMetricValueEveryParamSet += [list(x.keys())[0] for x in otherMetricResults]
                
                listMetricValueEveryParamSet = \
                    _write_to_np_MetricValueEveryParamSet(listMetricValueEveryParamSet, 
                                                          colnamesMetricValueEveryParamSet, 
                                                          j, epm[j], metricForCV, otherMetricResults)               
        
        for i in range(len(epm)):
            for j in range((-1-nOtherMetrics), 0):
                listMetricValueEveryParamSet[i][j] /= nFolds
        
        metricValues = numpy.array([x[(-1-nOtherMetrics):] for x in listMetricValueEveryParamSet])
        
        if eva.isLargerBetter():
            if nOtherMetrics > 0:
                self.bestIndex = numpy.argmax(metricValues[:,0])
            else:
                self.bestIndex = numpy.argmax(metricValues)
        else:
            if nOtherMetrics > 0:
                self.bestIndex = numpy.argmin(metricValues[:,0])
            else:
                self.bestIndex = numpy.argmin(metricValues)
        
        #return the best model
        self.bestModel = est.fit(dataset, epm[self.bestIndex])
        # convert list to pyspark.sql.DataFrame
        df = SQLContext.getOrCreate(SparkContext.getOrCreate()).createDataFrame(listMetricValueEveryParamSet, colnamesMetricValueEveryParamSet)
        df = df.withColumn(df.columns[0], df[df.columns[0]].cast(IntegerType()))
        return CrossValidatorModel(self.bestModel, df)
        
    #returns the hyperparameters of the best model chosen by the cross validator
    def getBestModelParams(self):
        epm = self.getOrDefault(self.estimatorParamMaps)
        
        if (self.bestIndex is not None):
            bestModelParms = dict((key.name, value) for key, value in epm[self.bestIndex].iteritems())
        else:
            bestModelParms = "\nCrossvalidation has not run yet.\n"
        return bestModelParms       
    
        
import unittest
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

def localRoundMetricValue(rr, key):
    rrAsDict = rr.asDict()
    rrAsDict[key] = round(rrAsDict[key], 3)
    return rrAsDict

class CrossValidatorWithStratificationIDTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession\
                   .builder\
                   .appName("CrossValidatorWithStratificationIDTests")\
                   .getOrCreate()
        dataFileName = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/datasetWithFoldID.csv"
        cls.data = cls.spark\
                  .read\
                  .option("header", "true")\
                  .option("inferSchema", "true")\
                  .csv(dataFileName)        
        cls.data.cache()
        
        
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        
    def test_CVResult(self):
        assembler = VectorAssembler(inputCols=self.data.columns[1:(-1)], outputCol="features")
        stratifyCol = "foldID"
        featureAssembledData = assembler.transform(self.data).select("y", "features", stratifyCol)   
        lr = LinearRegression(maxIter=1e5, standardization=True, 
                              featuresCol="features", labelCol="y")
        evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y")
        lambdas = [0.1,1]
        alphas = [0,0.5]
        paramGrid = ParamGridBuilder()\
               .addGrid(lr.regParam, lambdas)\
               .addGrid(lr.elasticNetParam, alphas)\
               .build()   
        validator = CrossValidatorWithStratificationID(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              stratifyCol=stratifyCol)
        cvModel = validator.fit(featureAssembledData)
        metrics = cvModel.avgMetrics.drop("paramSetID")
        collectedMetrics = metrics.collect()        
        roundedMetrics = [localRoundMetricValue(x, "metric for CV rmse") for x in collectedMetrics]
        
        self.assertEqual(len(roundedMetrics), 4, "Incorrect number of returned metric values.")
        expectedMetricStructure = [\
            {"regParam":0.1, "elasticNetParam": 0.0, "metric for CV rmse": 1.087},
            {"regParam":1.0, "elasticNetParam": 0.0, "metric for CV rmse": 1.052},
            {"regParam":0.1, "elasticNetParam": 0.5, "metric for CV rmse": 1.06},
            {"regParam":1.0, "elasticNetParam": 0.5, "metric for CV rmse": 1.069}\
        ]
        for metric in roundedMetrics:
            self.assertTrue(metric in expectedMetricStructure, 
                            "{0} is not expected. The expected {1}.".format(metric, expectedMetricStructure))
        
        bestParams = validator.getBestModelParams()
        self.assertEqual(bestParams, {'regParam': 1, 'elasticNetParam': 0}, "Incorrect best parameters.")
        
    def test_OutputNonNumericalGridSearch(self):
        assembler = VectorAssembler(inputCols=self.data.columns[1:(-1)], outputCol="features")
        stratifyCol = "foldID"
        featureAssembledData = assembler.transform(self.data).select("y", "features", stratifyCol)
        rf = RandomForestRegressor(featuresCol="features", 
                               labelCol="y",
                               minInstancesPerNode=1)
        evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="y")
        strategyGrid = ["sqrt", "5"]
        depthGrid = [3, 15]
        paramGrid = ParamGridBuilder()\
               .addGrid(rf.maxDepth, depthGrid)\
               .addGrid(rf.featureSubsetStrategy, strategyGrid)\
               .build()    
        validator = CrossValidatorWithStratificationID(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              stratifyCol=stratifyCol)
        cvModel = validator.fit(featureAssembledData)
        metrics = cvModel.avgMetrics.drop("paramSetID")
        collectedMetrics = metrics.collect()        
        roundedMetrics = [localRoundMetricValue(x, "metric for CV rmse") for x in collectedMetrics]
        
        self.assertEqual(len(roundedMetrics), 4, "Incorrect number of returned metric values.")
        expectedMetricStructure = [\
            {'metric for CV rmse': 1.073, 'maxDepth': 3.0, 'featureSubsetStrategy': 'sqrt'},
            {'metric for CV rmse': 1.07, 'maxDepth': 3.0, 'featureSubsetStrategy': '5'},
            {'metric for CV rmse': 1.111, 'maxDepth': 15.0, 'featureSubsetStrategy': 'sqrt'},
            {'metric for CV rmse': 1.108, 'maxDepth': 15.0, 'featureSubsetStrategy': '5'}\
        ]
        for metric in roundedMetrics:
            self.assertTrue(metric in expectedMetricStructure, 
                            "{0} is not expected. The expected {1}.".format(metric, expectedMetricStructure))
    
    def test_evaluateOtherMetrics(self):
        data_file_name = "s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/data_LRRFTemplat_FoldID.csv"
        data = CrossValidatorWithStratificationIDTests.spark\
                  .read\
                  .option("header", "true")\
                  .option("inferSchema", "true")\
                  .csv(data_file_name)        
        data = data.drop("OuterFoldID")
        data.cache()
        stratifyCol = "InnerFoldID"
        outcomeCol = "y"
        assembledFeatureCol = "features"        
        assembler = VectorAssembler(inputCols=data.columns[1:(-2)], outputCol=assembledFeatureCol)
        featureAssembledData = assembler.transform(data).select(outcomeCol, assembledFeatureCol, stratifyCol)
        lambdas = [0.1, 1]
        alphas = [0, 0.5]
        lr = LogisticRegression(maxIter=1e5, featuresCol = assembledFeatureCol,
                                labelCol = outcomeCol, standardization = False)
        evaluator = BinaryClassificationEvaluatorWithPrecisionAtRecall(rawPredictionCol="probability",
                                                  labelCol=outcomeCol)
        paramGrid = ParamGridBuilder()\
                   .addGrid(lr.regParam, lambdas)\
                   .addGrid(lr.elasticNetParam, alphas)\
                   .build()
        
        otherMetricSets = [{"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.05}},
                      {"metricName": "precisionAtGivenRecall", "metricParams": {"recallValue": 0.1}}]
        validator = CrossValidatorWithStratificationID(\
                        estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        stratifyCol=stratifyCol,
                        evaluateOtherMetrics=True, otherMetrics=otherMetricSets\
                    )
        cvModel = validator.fit(featureAssembledData)
        result_list = cvModel.avgMetrics.collect()
        result_list_of_dict = [x.asDict() for x in result_list]
        
        expected = [{"paramSetID":0,"elasticNetParam":0.0,"regParam":0.1,"metric for CV areaUnderROC":0.871,
                     "precisionAtGivenRecall at recallValue 0.05":0.963,"precisionAtGivenRecall at recallValue 0.1":0.908},
                    {"paramSetID":1,"elasticNetParam":0,"regParam":1,"metric for CV areaUnderROC":0.754,
                     "precisionAtGivenRecall at recallValue 0.05":0.776,"precisionAtGivenRecall at recallValue 0.1":0.732},
                    {"paramSetID":2,"elasticNetParam":0.5,"regParam":0.1,"metric for CV areaUnderROC":0.861,
                     "precisionAtGivenRecall at recallValue 0.05":0.963,"precisionAtGivenRecall at recallValue 0.1":0.845},
                    {"paramSetID":3,"elasticNetParam":0.5,"regParam":1,"metric for CV areaUnderROC":0.670,
                     "precisionAtGivenRecall at recallValue 0.05":0.697,"precisionAtGivenRecall at recallValue 0.1":0.699}]
        
        for result_metric_dict in result_list_of_dict:
            expected_metric_dict = None
            for m in expected:
                if (m["elasticNetParam"] == result_metric_dict["elasticNetParam"]) and \
                    (m["regParam"] == result_metric_dict["regParam"]):
                    expected_metric_dict = m
                    break
            self.assertTrue(expected_metric_dict is not None, "Result metric {} is not in the expected.".format(result_metric_dict)) 
            for key in result_metric_dict.keys():
                if key in ["paramSetID", "elasticNetParam", "regParam"]:
                    continue
                comparison = abs(result_metric_dict[key]-expected_metric_dict[key]) < 0.01
                self.assertTrue(comparison, "Result metric {0} is too different from the expected {1}.".format(result_metric_dict, expected_metric_dict)) 
        
if __name__ == "__main__":
    
    unittest.main()
    
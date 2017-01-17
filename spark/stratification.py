import numpy as np
# from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, count, udf

import unittest

__all__ = ["AppendDataMatchingFoldIDs"]

def AppendDataMatchingFoldIDs(data, nFolds, matchCol, foldCol="foldID", colsToKeep=None, nDesiredPartitions=None):
    """
    Select columns and then append a column of stratified fold IDs to the 
    input DataFrame.
    
    Parameters
    ----------
    data: pyspark.sql.dataframe.DataFrame
        The input DataFrame.
    nFolds: integer
        The number of foldds to stratify.
    matchCol: string
        The name of the column about which rows are matched (e.g., one row 
        with the positive outcome is matched with a number of rows with the
        negative outcome)
    foldCol: string
        The name of the appended column of fold IDs. The default is "foldID". 
    colsToKeep: list of strings
        The list of column names to keep in the input DataFrame. If None 
        (default), all input columns are kept. 
    nDesiredPartitions: integer or None.
        The number of partitions in the returned DataFrame. If "None",
        the result has the same number of partitions as the input data.
        Default: "None".
    
    Returns
    ----------
    A pyspark.sql.DataFrame with the following columns: columns indicated by 
    colsToKeep from the input DataFrame and the appended fold ID column. 
    """
    
    if colsToKeep is not None: 
        data = data.select(list(set(colsToKeep + [matchCol])))
    
    if foldCol in data.columns: 
        raise ValueError("foldCol {} already exists in the input data.".format(foldCol))
    
    if nDesiredPartitions is None:
        nDesiredPartitions = data.rdd.getNumPartitions()
    
    uniqueMatchIDs = data\
        .select(matchCol)\
        .distinct()\
        .collect()
    uniqueMatchIDs = [x.asDict()[matchCol] for x in uniqueMatchIDs]
    nPoses = len(uniqueMatchIDs)
    npFoldIDs = np.array(list(range(nFolds)) * np.ceil(float(nPoses) / nFolds))    
    npFoldIDs = npFoldIDs[:nPoses]    
    np.random.shuffle(npFoldIDs)
    result = SparkSession.builder.getOrCreate()\
        .createDataFrame([(int(npFoldIDs[i]),uniqueMatchIDs[i]) for i in range(nPoses)], [foldCol, matchCol])\
        .join(data, matchCol)\
        .coalesce(nDesiredPartitions)
    
    return result
    
def areFoldIdsEqual(ids):
    if len(set(ids)) == 1:
        return True
    else:
        return False

class StratificationTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName(cls.__name__).getOrCreate()
        cls.testData = cls.spark.read.option("header", "true")\
            .option("inferSchema", "true")\
            .csv("s3://emr-rwes-pa-spark-dev-datastore/lichao.test/data/toy_data/matchedID_label_features.csv")        
        cls.testData.cache()
        
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        
    def GetNumDistinctMatchedPosIDs(self, df):
        return df.select("foldID", "matched_positive_id")\
                        .rdd\
                        .groupByKey()\
                        .map(lambda x: len(set(x[1])))\
                        .collect()
    
    def test_Stratification(self):
        # 5 different matched_positive_ids
        # 2 folds
        nFolds = 2
        dataWithFoldID_2 = AppendDataMatchingFoldIDs(data=StratificationTests.testData, nFolds=nFolds, 
                                                     matchCol="matched_positive_id", colsToKeep=["matched_positive_id", "label", "features"])
        # those with the same matched_positive_id should have the same fold ID
        
        self.assertTrue(\
            dataWithFoldID_2.select("matched_positive_id", "foldID")\
                        .rdd\
                        .groupByKey()\
                        .map(lambda x: areFoldIdsEqual(list(x[1])))\
                        .reduce(lambda x,y: x&y),
            "Data with the same matched_positive_id should have the same fold ID!"
        )
        
        # one fold should have two different matched_positive_ids, the other 3
        nMatchedPosIDs = self.GetNumDistinctMatchedPosIDs(dataWithFoldID_2)
        self.assertTrue(\
            (nMatchedPosIDs == [2,3]) | (nMatchedPosIDs == [3,2]),
            "When split into 2 folds, one fold must have two distinct matched_positive_ids and the other must have 3."
        )
        
        # the union of the distinct matched_positive_ids should cover all those in the data
        setMatchedPositiveIDs = dataWithFoldID_2.select("foldID", "matched_positive_id")\
                        .rdd\
                        .groupByKey()\
                        .map(lambda x: set(x[1]))\
                        .reduce(lambda x,y: x.union(y))
        self.assertEqual(
            setMatchedPositiveIDs,
            set(StratificationTests.testData.select("matched_positive_id").rdd.map(lambda x: x.matched_positive_id).collect()),
            "The union of the distinct matched_positive_ids in all folds should cover all those in the data. "
        )
        
        # one matched_positive_id for each fold
        nFolds = 5
        dataWithFoldID_5 = AppendDataMatchingFoldIDs(data=StratificationTests.testData, nFolds=nFolds,
                                                     matchCol="matched_positive_id", colsToKeep=["matched_positive_id", "label", "features"])
        nMatchedPosIDs = self.GetNumDistinctMatchedPosIDs(dataWithFoldID_5)
        self.assertTrue(\
            nMatchedPosIDs == [1]*5,
            "When split into 5 folds, one fold must have only one distinct matched_positive_ids."
        )
        
        
        # two matched_positive_ids for each fold
        nFolds = 2
        dataWithFoldID_22 = \
            AppendDataMatchingFoldIDs(\
                data=StratificationTests.testData.filter(StratificationTests.testData.matched_positive_id != 0), 
                nFolds=nFolds,
                matchCol="matched_positive_id", 
                colsToKeep=["matched_positive_id", "label", "features"]
            )
        nMatchedPosIDs = self.GetNumDistinctMatchedPosIDs(dataWithFoldID_22)
        self.assertTrue(\
            nMatchedPosIDs == [2]*2,
            "Now the data has only 4 distinct values for matched_positive_ids. When split into 2 folds, each fold should have 2 distinct values."
        )
            
if __name__ == "__main__":
    
    unittest.main()
    
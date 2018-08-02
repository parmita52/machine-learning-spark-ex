from __future__ import print_function
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

import pandas as pd

"""Set up Spark connection"""
sc = SparkContext(master='local[*]', appName="KeithExample")
spark = SparkSession(sc)

"""Read in the data"""
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("Keith.csv")
df = df.withColumn("Bought_Flag", df["Bought"].cast("boolean").cast("int"))
df.show()
df.printSchema()

"""Elements of our Pipeline"""
color_indexer = StringIndexer(inputCol='Color', outputCol='Color_index')
assembler = VectorAssembler(inputCols=['Color_index', 'Size'], outputCol="feature_vector")
dt = DecisionTreeClassifier(labelCol="Bought_Flag", featuresCol="feature_vector")

"""Split data for training (70%) and testing (30%)"""
(trainingData, testData) = df.randomSplit([0.7, 0.3])

"""Pipeline: indexer (for colors), assembler (to vectorize), dt (decision tree)"""
pipeline = Pipeline(stages=[color_indexer, assembler, dt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)

"""Without a Pipeline"""
# x = (color_indexer.fit(trainingData)).transform(trainingData)
# x = assembler.transform(x)
# x = dt.fit(x)
# y = (color_indexer.fit(trainingData)).transform(testData)
# y = assembler.transform(y)
# predictions = x.transform(y)

predictions.show()

"""Analyze Results"""
def count_act_vs_pred(act, pred):
    return (predictions.filter((predictions.Bought_Flag == act) & (predictions.prediction == pred))).count()


true_positives = count_act_vs_pred(1, 1)
true_negatives = count_act_vs_pred(0, 0)
false_positives = count_act_vs_pred(0, 1)
false_negatives = count_act_vs_pred(1, 0)

tp_normal = true_positives / (true_positives + false_negatives)
fp_normal = false_positives / (false_positives + true_negatives)
tn_normal = true_negatives / (true_negatives + false_positives)
fn_normal = false_negatives / (false_negatives + true_positives)
total = predictions.count()
accuracy = (true_positives + true_negatives) / total

stats = {'Accuracy': accuracy,
         'TP': tp_normal,
         'FP': fp_normal,
         'TN': tn_normal,
         'FN': fn_normal}

pd.options.display.float_format = '{:.3f}'.format
stats_df = pd.Series(stats).to_frame()
print(stats_df)
dtm = model.stages[-1]
# dtm = x       \\without Pipeline
print(dtm.toDebugString)

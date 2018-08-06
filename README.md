# ML Spark Walkthrough Tutorial
This tutorial walks through a basic Machine Learning example from start to finish. It explains critical big ideas in ML and goes into specifics of how to use the Apache Spark ML Pipelines for implementation. 

If you already know the basics of ML and just want to learn about Spark Tools/Pipelines, feel free to skip to that section

- [General ML](https://github.com/parmita52/machine-learning-spark-ex#what-do-people-typically-use-ml-for)
- [Toy Example](https://github.com/parmita52/machine-learning-spark-ex#our-toy-example)
- [Picking a Model](https://github.com/parmita52/machine-learning-spark-ex#what-model-do-we-use)
- [Using the Model](https://github.com/parmita52/machine-learning-spark-ex#how-do-we-actually-use-it)
- [Spark Pipeline](https://github.com/parmita52/machine-learning-spark-ex#spark-pipeline)  <-- skip here if advanced 
- [Spark Tools](https://github.com/parmita52/machine-learning-spark-ex#spark-tools)
- [Implementing it!](https://github.com/parmita52/machine-learning-spark-ex#lets-get-to-it)
- [Results](https://github.com/parmita52/machine-learning-spark-ex#results)
- [Conclusion](https://github.com/parmita52/machine-learning-spark-ex#conclusion)
- [Future Steps](https://github.com/parmita52/machine-learning-spark-ex#where-to-go-from-here)
- [Bonus Tip](https://github.com/parmita52/machine-learning-spark-ex#bonus-tip-for-multiple-stringindexers)
- [Q & A](https://github.com/parmita52/machine-learning-spark-ex#q--a)


# What do people typically use ML for?
![alt_text](/images_n/chart1.png "Chart 1")
*Machine Learning*: A *machine* takes in (training) data. Then--without any explicit instructions--it *learns* from that data, usually resulting in a model that makes predictions for the future.

# Our toy example
![alt_text](/images_n/Keith1.jpg "Keith and knives")
Keith likes to collect knives and swords. There are 2 *features* he considers when deciding to get one: color and size  
Based on knives he bought in the past, can we predict whether or not he will buy a particular knife in the future?
![alt text](/images_n/chart2.png "Chart 2")
This dummy example I created can be found in `data.csv`. The data is only 50 rows long to allow for easy human understanding and intervention. Its purpose is just to help to learn the concepts of what's going on. In real life, data can be literally millions, if not billions, of rows long and thus it becomes basically impossible for humans to analyze on their own. 
Now’s a good time to open up the `data.csv` file in something like Excel to get a feel for what's going on.
![alt text](/images_n/goal2.png "Goal")

Cool, now that we have our data...

# What model do we use?
![alt text](/images_n/chart3.png "Chart 3")
There are tons of different models to use, so how do you pick which one? [Here](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/ "SAS ML Algo Cheat Sheet")'s a really good cheatsheet:

A lot of people hear the buzzword "Neural Nets" thrown around whenever Machine Learning is mentioned, but there are actually lots of different types of Machine Learning Algorithms--and depending on what you want, you'll pick a different method. 
Let's follow the cheat sheet for our example:

![alt text](/images_n/mlcheatsheet.png "Cheat Sheet")

**A.** We are not doing Dimension Reduction (that is sort of its entirely own topic you can read more about it [here](https://en.wikipedia.org/wiki/Dimensionality_reduction "Wikipedia"))

**B.** We do Have Responses aka we have supervised learning (a nice visual explanation [here](https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms "Nice Quora Answer") 
but in short, it basically means we have labelled training data. The past data we have on Keith's knife buying habits do have the label of whether or not he bought it. If we didn't have that “Bought” label in the data, we might be trying to do something else. Like maybe we'd look for other patterns in the data. For example “bigger knives tend to be blue” 
![alt text](/images_n/supervised.png "Supervised vs. Unsupervised")
Specifically, we are doing classification in this example. 

**C.** We are not predicting a numeric value. We are predicting whether he will buy the knife or not buy the knife. A numeric prediction might be something more like “How many knives will he buy?”

**D.** Since this is a beginner tutorial, I'm actually gonna go for the faster models (they also happen to be the simpler, more beginner-friendly ones)

**E.** I also want the data to be explainable, so I'll pick a decision tree (logistic regression could also be used for this example, but as I'll explain at the end, I sort of rigged this example with patterns to work well with the tree). In real life, in this situation, you may actually carry out both methods and compare them, since you don’t know which one will work better. 

Side note:
> One “unexplainable” method is actually the famous Neural Net! If the Neural Net predicted “Keith will not buy this knife”, it would not be able to clearly tell you why. With a Decision Tree, it will tell you “Keith will not buy this knife because its color/size...”
> Depending on what you want, this may or may not be useful, but for our purposes, I’d really like to see the explanation the algorithm comes up with. 
> When creating this data, I kept in mind that Keith doesn’t really like blue and he prefers larger knives over smaller ones. I want to see if the Decision Tree actually picks up on those patterns and is able to correctly say why certain things are the way they are. 

Awesome! Now we’ve settled on a Decision Tree :)  
You can read more about the math and theory behind Decision Trees [here](https://en.wikipedia.org/wiki/Decision_tree_learning "Wikipedia"), but if you don't need to know that, feel free to simply look at the next picture, and you'll pretty much understand what a Decision Tree is.

![alt text](/images_n/parkTree.png "Park DT")

**Our goal is to make a tree like this, but for Keith’s knife-buying scenario.**  
Out features are Color and Size and we want to predict Buying the knife. 

Keep in mind that in real scenarios, you could have hundreds of features and you could generate a decision tree so incredibly massive, that it’s basically just for computers to use (not very human-readable). 
![alt text](/images_n/bigdt.png "rly big tree")
You could also choose to limit the size of tree to ensure that it is in fact, human-readable. In our case, with 2 measly features, we’ll end up with a nice tiny human-readable tree anyways. 

Now that we've picked which model we are using...

# How do we actually use it? 

**To train a Decision Tree, you need to give it one vector column that represents all of the features and one numerical column that represents the prediction label.**  

So we have to take the data that we have and turn it into something...more number-y for the Decision Tree to work with. So let’s carefully look at the data we have so we can get started. 

In fact, one of the first thing you should always do in any machine learning project is to actually look at the data and plan your steps. Data scientists actually typically spend more than half their time cleaning and preparing data, and only the remaining time for training models and further analysis. 

Luckily, the data I provided purposefully needs minimal cleaning and prep. Let's take a look:  
We need to somehow turn the prediction label into a number and turn the features we have into a vector. 
![alt text](/images_n/goal.png "Data Prep")

So we have a nice little task list:
### Task 1 - Turn the Prediction Label (Bought) into something numerical
![alt text](/images_n/task1.png "Task 1")
**Solution:** Just use casting: `.cast("boolean").cast("int")`
---
### Task 2 - Turn the Colors feature into a numerical index 
(Size is already numerical, so we're all set there) 
![alt text](/images_n/task2.png "Task 2")
**Solution:** Use a [StringIndexer](https://spark.apache.org/docs/latest/ml-features.html#stringindexer)
---
### Task 3 - Combine the features into one feature vector column  
![alt text](/images_n/task3.png "Task 3")
**Solution:** Use a [VectorAssembler](https://spark.apache.org/docs/2.1.0/ml-features.html#vectorassembler)
---
### Task 4 - Use this prepared data to train a Decision Tree Model 
![alt text](/images_n/task4.png "Task 4")
**Solution:** Use a [DecisionTreeClassifier](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassifier)

Ok, so what exactly is a StringIndexer, VectorAssembler, and DecisionTreeClassifier? These three Spark Tools are the Estimators and Transformers that will make up our Spark Pipeline. (Don't worry if you didn't understand any of those terms yet--we're about to define them!)  

So to understand how to use them, first we have to learn what a Spark Pipeline is.

# Spark Pipeline
In general, an ML Pipeline is simply a process--a set of steps done in order--to the data to facilitate creating and using the model.  
The Spark ML Pipelines are the most widely used implementation of this concept. 

Spark gives an in depth technical description of their Pipelines [here](https://spark.apache.org/docs/2.2.0/ml-pipeline.html#example-estimator-transformer-and-param "Spark docs"), but if you just understand the cheat sheet below, you should be good. 
![alt text](/images_n/Pipeline.png)

Now that you understand Estimators and Transformers, we can go back to figuring out what the Spark Tools: StringIndexer, VectorAssembler, and DecisionTreeClassifier do. 

# Spark Tools 
Make sure you understand what each tools is doing and how that helps us accomplish things on our tasklist. Note how Estimators and Trandformers act slightly differently. 
![alt text](/images_n/tools_new.png "Spark Tools")
*Make sure you can see the difference between the* `DecisionTreeClassifier` (Estimator) *and the* `DecisionTreeClassificationModel` (Transformer).

So in total, we will be using these Spark Tools on our training and testing data like so:
![alt text](/images_n/chart5.png "Chart 5")
This way, we can use the training data to make our Decision Tree Model and then determine how good the model actually is by testing it against the testing data. 

Note that even though we will be splitting up into training and testing data, there is a lot of repeated action:  
*BOTH* training and testing data go through the String Indexer and the Vector Assembler so that they get turned into numbers.   
*ONLY TRAINING* data is used to generate the Decision Tree Model  
*ONLY TESTING* data is used with the Model to generate predictions  
 
Pipelines take care of this for you, so you avoid repeating code which often causes errors.  

If you still are a bit hazy on what "Pipeline," "Estimator," "Transformer," or any of the Spark Tools mean, please read over the cheat sheets again, or click on the links which go to the official Spark documentation for a more in depth, technical explanation. 

All right! Time to get started with the code!

# Let's get to it!

> Note that you should have Python and Spark installed for this to work  
> Python [here](https://www.python.org/downloads/)  
> Apache Spark [here](https://spark.apache.org/downloads.html)  

**Follow along with the code in `DecisionTree.py` as you look at this diagram. This is the important part where you should spend most of your time! Feel free to rewrite the code on your own so you really get what’s going on!**
![alt text](/images_n/chart6.png)

Open up `exampleOutput.txt` to see an example of what the code does accomplishes in the end. 

And that’s about it! You have just successfully used an ML Pipeline to carry out the training and testing of a Decision Tree! Congrats! :)

Note how much using the Pipeline radically improves the code:

Without a Pipeline:
```python
x = (color_indexer.fit(trainingData)).transform(trainingData)
x = assembler.transform(x)
x = dt.fit(x)
y = (color_indexer.fit(trainingData)).transform(testData)
y = assembler.transform(y)
predictions = x.transform(y)
```
With the Pipeline: 
```python
pipeline = Pipeline(stages=[color_indexer, assembler, dt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
```

Without pipelines, this is can be pretty difficult to follow. Most importantly however, we avoid repeated code! If later on, we decide there's actually one more step to modifying and preparing the data, without a pipeline, we would have to write it twice and make sure it properly lines up for training and testing. With a Pipeline, you simply add the extra step to the stages of the pipeline and voila! it does it correctly for both training and testing. 

Pipelines also take care of the difference between Estimators and Transformers for you so that you don’t have to worry about when to use `.fit().transform()` and when to use `.transform()`.

Pipelines also have the added benefit that they can be [saved](https://spark.apache.org/docs/2.2.0/ml-pipeline.html#saving-and-loading-pipelines) for later use. 

# Results

Phew! That was a lot of work! Time to look at our results. I have chosen to use accuracy stats with a confusion matrix, since they are among the most intuitive, but there are many other valid (and some more informative) metrics to use to judge the performance of your model (ROC score, sensitivity, specificity, etc.)

The model I generated looks like this (`exampleOutput.txt`):
![alt text](/images_n/finalTree.png "Yay! Final Tree!")
(Keep in mind that yours may look different depending on how the random split between your training and testing data went)

And it does really well:
![alt text](/images_n/confusionMatrix.png)
(How to interpret a confusion matrix [here](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/))

Now imagine we were given another new set of data, but this is unlabeled. We can again just use `model.transform(newdata)` and it will generate new predictions. 
In other words: 
If we walked into a store full of knives and we could see their color and size, we want to predict whether Keith would buy them or not without him telling us. We could pick up a knife, look at its color, measure its size and then using the tree, we could predict with 94% accuracy whether Keith would buy it or not. 

Normally, with such a small dataset, you wouldn't get such high accuracy ratings. But as I mentioned earlier, I somewhat rigged the data. As I was generating this data, I kept in mind that Keith hates blue knives, and generally prefers larger knives. The decision tree actually picked up on that underlying pattern correctly when it created its model! 
![alt_text](/images_n/Keith2.jpg "Keiths")
Think about that for a second! Isn’t it amazing!? I never once explicitly wrote in the code that “btw, I set up the data so that Keith doesn’t like blue, and prefers larger knives.” But by running this algorithm on the data, it automatically picked up on this pattern. That’s the power of machine learning!

# Conclusion 
Awesome! You just learned:
 - How to pick an ML algorithm 
 - The huge role data cleaning/prep plays in ML
 - Some techniques for preparing the data
 - What an ML Pipeline is
 - How to implement one using Spark 

Big Idea: How the general ML paradigm gets adapted to a specific use case
![alt text](/images_n/Paradigm.png "Paradigm")

# Where to go from here?
 - Read about more ML topics, like all the ones on that cheat sheet
 - Saw some words you didn’t recognize? ([clustering](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68), [association](https://en.wikipedia.org/wiki/Association_rule_learning), [ROC](https://www.dataschool.io/roc-curves-and-auc-explained/), [regression](https://www.quora.com/What-is-regression-in-machine-learning) etc…) Google them! Learn some more!
 - Improve this existing model by tuning the hyperparameters (ex. change the depth of this tree)
 - Test this out on a much larger, more legit dataset 
 - Use a completely different model and see if it does better or worse. (go ahead and try that Neural Net or maybe Logistic Regression or a Random Forest!)
 - Check out all the [other Transformers and Estimators](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#) Spark provides

# Bonus! Tip for multiple StringIndexers
Here we had exactly 1 categorical variable which we needed to assign an index to (Color). But imagine you had like fifty, or a hundred. It seems pretty bad to create an indexer for each and every single one of them like this:
```python
feature1_indexer = StringIndexer(inputCol='feature1', outputCol='feature1_index')
feature2_indexer = StringIndexer(inputCol='feature2', outputCol='feature2_index')
feature3_indexer = StringIndexer(inputCol='feature3', outputCol='feature3_index')
feature4_indexer = StringIndexer(inputCol='feature4', outputCol='feature4_index')
feature5_indexer = StringIndexer(inputCol='feature5', outputCol='feature5_index')
assembler = VectorAssembler(inputCols=['feature1_index', 'feature2_index', 'feature3_index', 'feature4_index', 'feature5_index'], outputCol="feature_vector")
dt = DecisionTreeClassifier(labelCol="prediction_label", featuresCol="feature_vector")
Pipeline = Pipeline(stages=[feature1_indexer, feature2_indexer, feature3_indexer, feature4_indexer, feature5_indexer,  assembler, dt])
```
Instead, you can just do
```python
categorical_features = [feature1, feature2, feature3, feature4, feature5]
indexers = [StringIndexer(inputCol=e, outputCol=e + “_index”).fit(df) for e in categorical_features] 
features_for_vector = [e + “_index” for e in categorical_features]
assembler = VectorAssembler(inputCols=features_for_vector, outputCol="feature_vector")
dt = DecisionTreeClassifier(labelCol="prediction_label", featuresCol="feature_vector")
Pipeline = Pipeline(stages= indexers + [assembler, dt])
```

Python list comprehension can be super useful here! Note how there’s less repeated code and fewer opportunities for error. Also, if I decide to add one more feature, I just have to add it in one location everything else automatically adjusts. 

# Q & A


















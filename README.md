# ML Spark Walkthrough Tutorial
This tutorial walks through a basic Machine Learning example from start to finish. It explains critical big ideas in ML and goes into specifics of how to use the Apache Spark ML Pipelines for implementation. 

If you already know the basics of ML and just want to learn about Spark Pipelines, feel free to skip to the Spark Pipeline Explanation Section. 

## What do people typically use ML for?
![alt_text](/images_n/chart1.png "Chart 1")
*Machine Learning*: A *machine* takes in (training) data. Then--without any explicit instructions--it *learns* from that data, usually resulting in a model that makes predictions for the future.

## Our toy example
Keith likes to collect knives. There are 2 features of knives he's looking at: color and size
Based on knives he bought in the past, can we predict whether or not he will buy a particular knife in the future?
![alt text](/images_n/chart2.png "Chart 2")

This dummy example I created can be found in data.csv. The data is only 50 rows long to allow for easy human understanding and intervention. It's purpose to to help to learn the concepts of what's going on. In real life, data can be literally millions, if not billions, of rows long and thus it becomes basically impossible for humans to analyze on their own. 
Now’s a good time to open up the .csv file in something like Excel to get a feel for what's going on.
![alt text](/images_n/goal2.png "Goal")

Cool, now that we have our data...

## What model do we use?
![alt text](/images_n/chart3.png "Chart 3")
There are tons of different models to use, so how do you pick which one? [Here](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/ "SAS ML Algo Cheat Sheet")'s a really good cheatsheet:

A lot of people hear the buzzword "Neural Nets" thrown around whenever Machine Learning is mentioned, but there are actually lots of different types of Machine Learning Algorithms--and depending on what you want, you'll pick a different method. 
Let's follow the cheat sheet for our example:

![alt text](/images_n/cheatsheet.png "Cheat Sheet")

A. We are not doing Dimension Reduction (that is sort of its entirely own topic you can read more about it [here](https://en.wikipedia.org/wiki/Dimensionality_reduction "Wikipedia"))

B. We do Have Responses aka we have supervised learning (a nice visual explanation [here](https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms "Nice Quora Answer") 
but in short, it basically means we have labelled training data. The past data we have on Keith's knife buying habits do have the label of whether or not he bought it. If we didn't have that “Bought” label in the data, we might be trying to do something else. Like maybe we'd look for other patterns in the data. For example “bigger knives tend to be blue” 
![alt text](/images_n/supervised.png "Supervised vs. Unsupervised")
Specifically, we are doing classification in this example. 

C. We are not predicting a numeric value. We are predicting whether he will buy the knife or not buy the knife. A numeric prediction might be something more like “How many knives will he buy?”

D. Since this is a beginner tutorial, I'm actually gonna go for the faster models (they also happen to be the simpler, more beginner-friendly ones)

E. I also want the data to be explainable, so I'll pick a decision tree (logistic regression could also be used for this example, but as I'll explain at the end, I sort of rigged this example with patterns to work well with the tree). In real life, in this situation, you may actually carry out both methods and compare them, since you don’t know which one will work better. 

Side note:
> One “unexplainable” method is actually the famous Neural Net! If the Neural Net predicted “Keith will not buy this knife”, it would not be able to clearly tell you why. With a Decision Tree, it will tell you “Keith will not buy this knife because its color/size...”
> Depending on what you want, this may or may not be useful, but for our purposes, I’d really like to see the explanation the algorithm comes up with. 
> When creating this data, I kept in mind that Keith doesn’t really like blue and he prefers larger knives over smaller ones. I want to see if the Decision Tree actually picks up on those patterns and is able to correctly say why certain things are the way they are. 

Awesome! Now we’ve settled on a Decision Tree :)
You can read more about the math and theory behind Decision Trees [here](https://en.wikipedia.org/wiki/Decision_tree_learning "Wikipedia"), but if you don't need to know that, feel free to simply look at the next picture, and you'll pretty much understand what a Decision Tree is.

![alt text](/images_n/parkTree.png "Park DT")

Our goal is to make a tree like this, but for Keith’s knife-buying scenario. 
Out features are Color and Size and we want to predict Buying the knife. 

Keep in mind that in real scenarios, you could have hundreds of features and you could generate a decision tree so incredibly massive, that it’s basically just for computers to use (not very human-readable). 
![alt text](/images_n/bigtree.png "rly big tree")
You could also choose to limit the size of tree to ensure that it is in fact, human-readable. In our case, with 2 measly features, we’ll end up with a nice tiny human-readable tree anyways. 

Now that we've picked which model we are using...

## How do we actually use it? 

**To train a Decision Tree, you need to give it one vector column that represents all of the features and one numerical column that represents the prediction label. **

So we have to take the data that we have and turn it into something...more number-y for the Decision Tree to work with. So let’s carefully look at the data we have so we can get started. 

In fact, one of the first thing you should always do in any machine learning project is to actually look at the data and plan your steps. Data scientists actually typically spend more than half their time cleaning and preparing data, and only the remaining time for training models and further analysis. 

Luckily, the data I provided purposefully needs minimal cleaning and prep. Let's take a look:
We need to somehow turn the features we have into a vector and have the prediction be number. 
![alt text](/images_n/goal.png "Data Prep")

So we have a nice little task list:
###### Task 1 - turn the Prediction Label (Bought) into something numerical
![alt text](/images_n/task1.png "Task 1")
###### Task 2 - Turn the Colors feature into a numerical index 
(Size is already numerical, so we're all set there) 
![alt text](/images_n/task2.png "Task 2")
###### Task 3 - Combine the features into one features vector column  
![alt text](/images_n/task3.png "Task 3")
###### Task 4 - Use this prepared data to train a Decision Tree Model 
![alt text](/images_n/task4.png "Task 4")

How will we accomplish each of these tasks? Spark provides a lot of tools for exactly this purpose. In fact, Task 1 is super simple and Tasks 2, 3, and 4 have special Spark Tools made just for their purpose.

###### Task 1
Just use casting (actually pretty easy, does not need ML tools)
`df = df.withColumn("Bought_Flag", df["Bought"].cast("boolean").cast("int"))`

###### Task 2
Use a [StringIndexer](https://spark.apache.org/docs/latest/ml-features.html#stringindexer)
###### Task 3
Use a [VectorAssembler](https://spark.apache.org/docs/2.1.0/ml-features.html#vectorassembler)
###### Task 4
Use a [DecisionTreeClassifier](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassifier)

Ok so what exactly are these three new...

## Spark Tools 
For now, read the sheet below and just understand what each tool does. The rest (about Estimators and Transformers) will make more sense after reading the next sheet about Pipelines, Estimators, and Transformers. 
![alt text](/images_n/tools.png "Spark Tools")










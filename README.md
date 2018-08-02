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

B. We do Have Responses aka we have supervised learning (a nice visual explanation here [Link](https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms "Nice Quora Answer") 
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



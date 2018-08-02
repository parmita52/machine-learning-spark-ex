# ML Spark Walkthrough Tutorial
This tutorial walks through a basic Machine Learning example from start to finish. It explains critical big ideas in ML and goes into specifics of how to use the Apache Spark ML Pipelines for implementation. 

If you already know the basics of ML and just want to learn about Spark Pipelines, feel free to skip to the Spark Pipeline Explanation Section. 

## What do people typically use ML for?
![alt text](https://github.com/parmita52/machine-learning-spark-ex/blob/master/images_n/chart1.png "Chart 1")
*Machine Learning*: A *machine* takes in (training) data. Then--without any explicit instructions--it *learns* from that data, usually resulting in a model that makes predictions for the future.

## Our toy example
Keith likes to collect knives. There are 2 features of knives he's looking at: color and size
Based on knives he bought in the past, can we predict whether or not he will buy a particular knife in the future?
![alt text](https://github.com/parmita52/machine-learning-spark-ex/blob/master/images_n/chart2.png "Chart 2")

This dummy example I created can be found in data.csv. The data is only 50 rows long to allow for easy human understanding and intervention. It's purpose to to help to learn the concepts of what's going on. In real life, data can be literally millions, if not billions, of rows long and thus it becomes basically impossible for humans to analyze on their own. 
Now’s a good time to open up the .csv file in something like Excel to get a feel for what's going on.
![alt text](https://github.com/parmita52/machine-learning-spark-ex/blob/master/images_n/goal2.png "Goal")

Cool, now that we have our data...

## What model do we use?
![alt text](https://github.com/parmita52/machine-learning-spark-ex/blob/master/images_n/chart3.png "Chart 3")
There are tons of different models to use, so how do you pick which one? Here's a really good cheatsheet:
[Link](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/ "SAS ML Algo Cheat Sheet")

A lot of people hear the buzzword "Neural Nets" thrown around whenever Machine Learning is mentioned, but there are actually lots of different types of Machine Learning Algorithms--and depending on what you want, you'll pick a different method. 
Let's follow the cheat sheet for our example:
![alt text](https://github.com/parmita52/machine-learning-spark-ex/blob/master/images_n/cheatsheet.png "Cheat Sheet")


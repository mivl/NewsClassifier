News Classifier
===============

Reads input news in XML format and trains a model for classifying news into different categories (sports, literature etc) according to their content (word frequency). Some text preprocessing steps like tokenization and stop words removal are applied, and then two classification algorithms (Naive Bayes and Decision Tree) and used to train a classifier and the results are compared.

Code was written in Scala (v. 2.11.8) using the Apache Spark (v. 2.0.1) framework.

Requires an external jar file for stemming (disabled).

Running
-------

1. Go to https://spark.apache.org/downloads.html and download Apache Spark 2.0.1
2. Go to the project directory and run the following command:
path/to/spark-submit --jars spark-stemming-0.1.1.jar --class Main newsclassifier_2.11-1.0.jar

import org.apache.spark.ml.feature._
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable
import scala.xml.XML

/**
  * Created by isabellima on 16/11/2016.
  */
object Main {
  def main(arg: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder()
      .appName("NewsClassifier")
      .getOrCreate()

    import spark.implicits._

    val file = XML.loadFile("news_data.xml")
    val categories = (file \ "category" \ "name").map(x => x.text).toList

    val news = (file \ "item").map(x => ((x \ "@category").text, (x \ "title").text + " "
      + (x \ "description").text + " " + (x \ "text").text)).toDF("category", "text")

    //Tokenization
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("[^\\wÀ-ú]+")

    val tokenized = tokenizer.transform(news)

    //Stop words removal
    val stopWords = StopWordsRemover.loadDefaultStopWords("portuguese")
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")
      .setStopWords(stopWords :+ "é" :+ "r")

    val removed = remover.transform(tokenized)

    //Stemming
    val stemmer = new Stemmer()
      .setInputCol("filtered")
      .setOutputCol("stemmed")
      .setLanguage("Portuguese")

    /*val r2 = removed.rdd.map(x => (x.get(0), x.get(1), x.get(2),
      x.get(3).asInstanceOf[mutable.WrappedArray[String]].flatMap(x => x))).toDF("category","text","tokens","filtered")
    val stemmed = stemmer.transform(r2)*/

    //Preparing for classification
    val numOfFeatures = 100

    val allWords = removed.select("filtered").rdd.map(x => x(0).asInstanceOf[mutable.WrappedArray[String]]).flatMap(x => x)
    val topWords = allWords.map(x => (x,1)).reduceByKey((x,y) => x+y).map(x => (x._2, x._1)).top(numOfFeatures).map(x => x._2)

    val categoriesMap = categories.zipWithIndex.toMap

    val labelsAndFeatures = removed.rdd.map(x => (x.get(0).asInstanceOf[String], Vectors.dense(topWords
      .map(y => if (x.get(3).asInstanceOf[mutable.WrappedArray[String]].toArray.contains(y)) 1.0 else 0.0))))

    val data = labelsAndFeatures.toDF("label","features")

    // Fit on whole dataset to include all labels in index
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Automatically identify categorical features, and index them
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(data)

    // Split the data into training and test sets
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    /*** DECISION TREE CLASSIFIER ***/
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val DTpipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val DTmodel = DTpipeline.fit(trainingData)
    val DTpredictions = DTmodel.transform(testData)

    DTpredictions.select("predictedLabel", "label", "features").show(5)

    val DTevaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val DTaccuracy = DTevaluator.evaluate(DTpredictions)
    println("Decision Tree accuracy = " + DTaccuracy)

    /*** NAIVE BAYES CLASSIFIER ***/
    val nb = new NaiveBayes()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    val NBpipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, nb, labelConverter))

    val NBmodel = NBpipeline.fit(trainingData)
    val NBpredictions = NBmodel.transform(testData)

    NBpredictions.select("predictedLabel", "label", "features").show(5)

    val NBevaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val NBaccuracy = NBevaluator.evaluate(NBpredictions)
    println("Naive Bayes accuracy: " + NBaccuracy)
  }
}
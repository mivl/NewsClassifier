name := "NewsClassifier"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1" % "provided"

libraryDependencies += "org.scala-lang" % "scala-xml" % "2.11.0-M4"

libraryDependencies += "master" % "spark-stemming" % "0.1.1"
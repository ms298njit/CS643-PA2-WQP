FROM bde2020/spark-maven-template:3.3.0-hadoop3.3

MAINTAINER Mehul Shah

ENV SPARK_APPLICATION_MAIN_CLASS winequality.WineQualityPrediction

ENV SPARK_APPLICATION_JAR_NAME WQP-1.0

ENV SPARK_APPLICATION_ARGS "file:///opt/workspace/ValidationDataset.csv file:///opt/workspace/model"

VOLUME /opt/workspace



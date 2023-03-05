"""
This is the Python module containing the functions for the data pipeline
used for the Medium post

Author: Jose D. Hernandez-Betancur
Date: 2023-03-04
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import os
import contextlib
from prefect import task
from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import udf, col, size, min as spark_min
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import (StopWordsRemover, Tokenizer, NGram,
                                HashingTF, MinHashLSH, RegexTokenizer,
                                SQLTransformer)


@contextlib.contextmanager
def get_spark_session(conf: SparkConf):
    """
    Function that is wrapped by context manager

    Args:
      - conf(SparkConf): It is the configuration for the Spark session
    """

    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    try:
        yield spark
    finally:
        spark.stop()


@task(name="Loading GroupLens movies")
def load_movie_data(spark: SparkSession, dir_path: str) -> SparkDataFrame:
    """
    Function to load the GroupLens movies

    Args:
      - spark (SparkSession): The spark session
      - dir_path (string): It is the path for the main.py
    Output:
      - movies_ddf (SparkDataFrame): Spark dataframe containing the GroupLens movies
    """

    strip_year_udf = udf(lambda title: title[:-7])

    file_path = os.path.join(dir_path,
                        'input_data',
                        'movies.csv')

    movies_ddf = (spark.read.csv(file_path, header=True, inferSchema=True)
              .drop('genres')
              .withColumn('Title', strip_year_udf(col('title'))))

    return movies_ddf


@task(name="Loading GroupLens rating")
def load_rating_data(spark: SparkSession, dir_path: str) -> SparkDataFrame:
    """
    Function to load the GroupLens rating

    Args:
      - spark (SparkSession): The spark session
      - dir_path (string): It is the path for the main.py
    Output:
      - ratings_ddf (SparkDataFrame): Spark dataframe containing the GroupLens rating
    """

    file_path = os.path.join(dir_path,
                        'input_data',
                        'ratings.csv')

    ratings_ddf = (spark.read.csv(file_path, header=True, inferSchema=True)
                .drop('timestamp'))

    return ratings_ddf


@task(name="Average rating for the movies")
def average_rating(movies_ddf: SparkDataFrame,
          ratings_ddf: SparkDataFrame) -> SparkDataFrame:
    """
    Function to calculate the average rating per movie and join the two DDF

    Args:
      - movies_ddf (SparkDataFrame): Lens movies
      - ratings_ddf (SparkDataFrame): Lens rating

    Output:
      - lens_ddf (SparkDataFrame): Dataframe after joining ratings and movies
    """

    lens_ddf = (ratings_ddf
        .groupby('movieId')
        .avg('rating')
        .select(col('movieId'), col('avg(rating)').alias('Rating'))
        .join(movies_ddf, 'movieId'))

    return lens_ddf


@task(name="Training the data preprocessing pipeline")
def prepocessing_data(lens_ddf: SparkDataFrame) -> PipelineModel:
    """
    Function to train the data preprocessing pipepeline

    Agrs:
      - lens_ddf (SparkDataFrame): Lens spark dataframe
    
    Otput:
      - model: (Pipeline): Trained model
    """

    model = Pipeline(stages=[
    SQLTransformer(statement="SELECT *, lower(Title) lower FROM __THIS__"),
    Tokenizer(inputCol="lower", outputCol="token"),
    StopWordsRemover(inputCol="token", outputCol="stop"),
    SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__"),
    RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1),
    NGram(n=2, inputCol="char", outputCol="ngram"),
    HashingTF(inputCol="ngram", outputCol="vector"),
    MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)
            ]).fit(lens_ddf)
    print('The data type is ', type(model))

    return model


@task(name="Loading IMDB sample movies")
def load_imbd_data(spark: SparkSession, dir_path: str) -> SparkDataFrame:
    """
    Function to load the IMDB data sample

    Args:
      - spark (SparkSession): The spark session
      - dir_path (string): It is the path for the main.py
    Output:
      - ratings_ddf (SparkDataFrame): Spark dataframe containing the IMDB sample movies
    """

    file_path = os.path.join(dir_path,
                        'input_data',
                        'imdb_sample.csv')

    imdb_df = (spark.read.csv(file_path, sep=';', header='true')
           .select('Id', 'Title', col('ImdbScore').alias('Rating')))

    return imdb_df


@task(name="Transform lens dataframe")
def transform_lens(model: PipelineModel,
                  lens_ddf: SparkDataFrame) -> SparkDataFrame:
    """
    Function to transform the lens dataframe for fuzzy matching

    Args:
      - model (Pipeline): Tuned model for fuzzy-based distance
      - lens_ddf (SparkDataFrame): Lens dataframe
    Output:
      - result_lens (SparkDataFrame): Dataframe after preprocessing lens data
    """

    result_lens = model.transform(lens_ddf)
    result_lens = result_lens.filter(size(col("ngram")) > 0)

    return result_lens


@task(name="Transform IMDB dataframe")
def transform_imdb(model: PipelineModel,
                  imdb_df: SparkDataFrame) -> SparkDataFrame:
    """
    Function to transform the IMDB dataframe for fuzzy matching

    Args:
      - model (Pipeline): Tuned model for fuzzy-based distance
      - imdb_df (SparkDataFrame): IMDB dataframe
    Output:
      - result_imdb (SparkDataFrame): Dataframe after preprocessing IMDB data
    """

    result_imdb = model.transform(imdb_df)
    result_imdb = result_imdb.filter(size(col("ngram")) > 0)

    return result_imdb


@task(name="Similarity join")
def similarity_join(result_imdb: SparkDataFrame,
                    result_lens: SparkDataFrame,
                    threshold: float,
                    model: PipelineModel) -> SparkDataFrame:
    """
    Function to apply similary join

    Agrs:
      - result_imdb (SparkDataFrame): Dataframe after preprocessing IMDB data
      - result_lens (SparkDataFrame): Dataframe after preprocessing lens data
      - threshold (float): Threshold for using with the fuzzy-logic-based distance
      - model (Pipeline): Tuned model for fuzzy-based distance
    Output:
      - result (SparkDataFrame): Resulting dataframe after similarity join
    """

    window_func = Window.partitionBy('datasetA.id')

    result = model.stages[-1].approxSimilarityJoin(result_imdb,
                                        result_lens,
                                        threshold,
                                        "jaccardDist")

    result = (result
              .withColumn('minDist', spark_min('jaccardDist').over(window_func))
              .where(col('jaccardDist') == col('minDist'))
              .withColumn('IMDB_Title', col('datasetA.Title'))
              .withColumn('Lens_Title', col('datasetB.Title'))
              .drop('minDist'))

    print(result
      .select('IMDB_Title', 'Lens_Title', 'jaccardDist')
      .sort(col('datasetA.id'))
      .show(5))

    result = result.select('IMDB_Title', 'Lens_Title',
                        col('datasetA.Rating').alias('IMDB_Rating'),
                        col('datasetB.Rating').alias('Lens_Ratin'))

    print(result.show(5))

    return result


@task(name="Saving the similarity join results")
def saving_results(result: SparkDataFrame,
                  dir_path: str,
                  n_executors: int):
    """
    Function to save the final results

    Args:
      - result (SparkDataFrame): dataframe containing the similarity join results
      - dir_path (string): It is the path for the main.py
      - n_executors (integer): number of executors
    """

    file_path = os.path.join(dir_path,
                        'output_data',
                        'final_result')

    result.coalesce(n_executors).write.mode("overwrite").parquet(file_path)

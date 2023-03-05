"""
This is the main module that is used for orchestrating the data pipeline
for the Medium post

Author: Jose D. Hernandez-Betancur
Date: 2023-03-04
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import os
import argparse
from multiprocessing import cpu_count
from prefect import flow
from pyspark import SparkConf

from pipeline_tasks import (get_spark_session,
                            load_rating_data,
                            load_movie_data,
                            load_imbd_data,
                            average_rating,
                            prepocessing_data,
                            transform_imdb,
                            transform_lens,
                            similarity_join,
                            saving_results)

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

@flow(name="Medium Data Pipeline Flow")
def data_pipeline(threshold: float):
    '''
    Function wrapped to be used as the Prefect main flow
    '''

    # CPU numbers
    n_cpus = cpu_count()
    n_executors = n_cpus - 1
    n_cores = 4
    n_max_cores = n_executors * n_cores

    # Add additional spark configurations
    conf = SparkConf().setMaster(f'local[{n_cpus}]').setAppName("Medium post")
    conf.set("spark.sql.legacy.parquet.int96RebaseModeInRead", "LEGACY")
    conf.set("spark.sql.legacy.parquet.int96RebaseModeInWrite", "LEGACY")
    conf.set("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "LEGACY")
    conf.set("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "LEGACY")
    conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    conf.set("parquet.enable.summary-metadata", "false")
    conf.set("spark.sql.broadcastTimeout",  "3600")
    conf.set("spark.sql.autoBroadcastJoinThreshold",  "1073741824")
    conf.set("spark.dynamicAllocation.enabled", "true")
    conf.set("spark.sql.debug.maxToStringFields", "100")
    conf.set("spark.executor.memory", "10g")
    conf.set("spark.driver.memory", "10g")
    conf.set("spark.executor.cores", str(n_cores))
    conf.set("spark.cores.max", str(n_max_cores))
    conf.set("spark.storage.memoryFraction", "0")
    conf.set("spark.driver.maxResultSize", "8g")
    conf.set("spark.files.overwrite","true")

    # Setting up the Spark cluster
    with get_spark_session(conf=conf) as spark_session:

        movies_ddf = load_movie_data(spark=spark_session,
                                      dir_path=dir_path)

        imdb_df = load_imbd_data(spark=spark_session,
                                  dir_path=dir_path)

        ratings_ddf = load_rating_data(spark=spark_session,
                                  dir_path=dir_path)

        lens_ddf = average_rating(movies_ddf=movies_ddf,
                                  ratings_ddf=ratings_ddf)

        model = prepocessing_data(lens_ddf=lens_ddf)

        result_lens = transform_lens(model=model,
                                  lens_ddf=lens_ddf)

        result_imdb = transform_imdb(model=model,
                                    imdb_df=imdb_df)

        result = similarity_join(result_imdb=result_imdb,
                                  result_lens=result_lens,
                                  threshold=threshold,
                                  model=model)

        saving_results(result=result,
                      dir_path=dir_path,
                      n_executors=n_executors)


if __name__ == '__main__':

    # Input argument to change the flow parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', '-t',
                        help='Threshold for using with the fuzzy-logic-based distance',
                        type=float,
                        required=False,
                        default=1.0)
    args = parser.parse_args()

    # Running the main flow
    data_pipeline(threshold=args.threshold)

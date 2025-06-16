#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, date_trunc, col, when, avg, to_timestamp, regexp_replace, broadcast
from pyspark.sql.types import IntegerType
import shutil

import os

spark = SparkSession.builder.appName("myApp").getOrCreate()

bike_folder_path = "./historical_datasets/Bike Historical Data"

weather_path = "./historical_datasets/Weather Historical Data/weather_data.csv"

bike_file_list = [os.path.join(bike_folder_path, f) for f in os.listdir(bike_folder_path) if f.endswith(".csv")] # Makes sure it's a CSV file

def main():
    weather_df = load_weather_data()
    weather_df = broadcast(weather_df)
    for file in bike_file_list:
        bike_df = load_bike_data(file)
        joined_data = bike_df.join(weather_df, "time")
        joined_data.write.mode("append").json("./processed_dataset/processed.json")
    produce_final_table()
    
def load_weather_data():
    weather_df = spark.read.csv(weather_path, header=True)
    weather_df = clean_weather_table(weather_df)
    return weather_df

def clean_weather_table(df):
    df = df.filter(col("HourlyDryBulbTemperature").isNotNull()).select("DATE", "HourlyDryBulbTemperature")
    df = truncate_time_weather(df)
    df = cast_temperature_as_integer(df)
    return df
    
def truncate_time_weather(df):
    df = df.withColumn("time", to_timestamp(df.DATE))
    df = df.withColumn("time", date_trunc("hour", "time"))
    df = df.select("HourlyDryBulbTemperature", "time")
    return df

def cast_temperature_as_integer(df):
    df = df.withColumn("HourlyDryBulbTemperature", regexp_replace("HourlyDryBulbTemperature",'s$',''))
    df = df.withColumn("HourlyDryBulbTemperature", col("HourlyDryBulbTemperature").cast(IntegerType()))
    return df.groupBy("time").agg(avg("HourlyDryBulbTemperature"))


def load_bike_data(file):
    df = spark.read.csv(file,header = True)
    df = remove_null_values_from_record(df)
    df = clean_bike_table(df)
    return df

def remove_null_values_from_record(df):
    # Filter entries with non-null station information
    df = df.filter(df.missing_station_information == False).select("station_id","num_bikes_available","num_ebikes_available","station_status_last_reported","capacity")

    # Changed bikes and ebikes from string to integer for calculation
    df = df.withColumn(
        "num_ebikes_available",
        remove_null_from_column("num_ebikes_available"))

    df = df.withColumn(
        "num_bikes_available",
        remove_null_from_column("num_bikes_available")
    )
    return df

def remove_null_from_column(column_name):
    return (when((col(column_name)) == "\\N", "0").otherwise(col(column_name))).cast(IntegerType())

def clean_bike_table(df):
    df = df.withColumn("availability", calculate_bike_availability(df))
    df = truncate_time_bike(df)
    df = df.select("station_id", "availability", "time")
    return df.groupBy("time", "station_id").agg(avg("availability"))

def calculate_bike_availability(df):
    availability = (df.num_bikes_available + df.num_ebikes_available) / df.capacity
    return when(df.capacity == 0, 1).when(availability > 1, 1).otherwise(availability)

def truncate_time_bike(df):
    df = df.withColumn("date", from_unixtime("station_status_last_reported"))
    df = df.withColumn("time", date_trunc("hour", "date"))
    return df
    # return availability



def produce_final_table():
    final_df = spark.read.json("./processed_dataset/processed.json")
    final_df.coalesce(1).write.json("./final_output/")

    shutil.rmtree("./processed_dataset")


    
    




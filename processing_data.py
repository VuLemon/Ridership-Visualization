
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, date_trunc, col, when, avg, to_timestamp, regexp_replace, broadcast, hour, date_format
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.functions import vector_to_array
import shutil
import pandas as pd
import sklearn
from pathlib import Path
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler, VectorAssembler, StringIndexer

import os

import re

import joblib


spark = SparkSession.builder.appName("myApp").getOrCreate()

bike_folder_path = "./historical_datasets/Bike Historical Data"

weather_path = "./historical_datasets/Weather Historical Data/weather_data.csv"

bike_file_list = [os.path.join(bike_folder_path, f) for f in os.listdir(bike_folder_path) if f.endswith(".csv")] # Makes sure it's a CSV file



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
    df = df.select("time","station_id", "availability")
    return df.groupBy("time", "station_id").agg(avg("availability"))

def calculate_bike_availability(df):
    availability = (df.num_bikes_available + df.num_ebikes_available) / df.capacity
    return when(df.capacity == 0, 1).when(availability > 1, 1).otherwise(availability)

def truncate_time_bike(df):
    df = df.withColumn("date", from_unixtime("station_status_last_reported"))
    df = df.withColumn("time", date_trunc("hour", "date"))
    return df
    # return availability



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
    return df

def cast_temperature_as_integer(df):
    df = df.withColumn("HourlyDryBulbTemperature", regexp_replace("HourlyDryBulbTemperature",'s$','')) #Some number ends in s, like 74s for some reason
    df = df.withColumn("HourlyDryBulbTemperature", col("HourlyDryBulbTemperature").cast(IntegerType()))
    df = df.groupBy("time").agg(avg("HourlyDryBulbTemperature"))
    df.printSchema()
    return df.groupBy("time").agg(avg("HourlyDryBulbTemperature"))



def process_bike_files(weather_df):
    for i, file in enumerate(bike_file_list):
        bike_df = load_bike_data(file)
        joined_data = bike_df.join(weather_df, on=["time"])
        joined_data.printSchema()
        joined_data.show(1, truncate=False)
        joined_data.write.mode("overwrite").option("header", True).csv(f"./processed_dataset/batch_{i}")


def save_processed_data(df):
    df = df.withColumn("hour", hour("time"))
    df = df.withColumn("weekday", date_format("time", "EEEE"))
    df = normalize_temperature(df)
    df = one_hot_encode_data(df)
    df.printSchema()
    df.drop("temp_vec").write.mode("overwrite").option("header", True).csv("./final_output/cleaned_data")


def normalize_temperature(df):
    df = df.withColumn("avg(HourlyDryBulbTemperature)", col("avg(HourlyDryBulbTemperature)").cast(DoubleType()))
    temp_assembler = VectorAssembler(inputCols=["avg(HourlyDryBulbTemperature)"], outputCol="temp_vec")
    vector_dataframe = temp_assembler.transform(df)

    # Apply MinMaxScaler
    scaler = MinMaxScaler(inputCol="temp_vec", outputCol="temperature_scaled")
    scaler_model = scaler.fit(vector_dataframe)

    # Step 2: Transform to get scaled vector column
    df_scaled = scaler_model.transform(vector_dataframe)

    # Step 3: Extract the first (and only) element of the scaled vector into a scalar column
    df_scaled = df_scaled.withColumn("temp_scaled", vector_to_array("temperature_scaled")[0])

    df_scaled = df_scaled.drop("temp_vec", "temperature_scaled")

    print("finished normalizing temperature")
    return df_scaled

def one_hot_encode_data(df):
    hour_indexer = StringIndexer(inputCol="hour", outputCol="hour_index").fit(df)
    weekday_indexer = StringIndexer(inputCol="weekday", outputCol="weekday_index").fit(df)
    station_indexer = StringIndexer(inputCol="station_id", outputCol="station_id_index").fit(df)

    df = hour_indexer.transform(df)
    df = weekday_indexer.transform(df)
    df = station_indexer.transform(df)

    # Save labels
    hour_labels = hour_indexer.labels
    weekday_labels = weekday_indexer.labels
    station_labels = station_indexer.labels

    # Step 2: OneHotEncoder
    encoder = OneHotEncoder(
        inputCols=["hour_index", "weekday_index", "station_id_index"],
        outputCols=["hour_vec", "weekday_vec", "station_vec"],
        dropLast= False  # <-- NOTE: output should be station_vec not station_id_vec
    )
    df_encoded = encoder.fit(df).transform(df)

    # Step 3: Expand one-hot vectors to individual columns
    df_encoded = expand_onehot(df_encoded, "hour_vec", hour_labels, "hour")
    df_encoded = expand_onehot(df_encoded, "weekday_vec", weekday_labels, "weekday")
    df_encoded = expand_onehot(df_encoded, "station_vec", station_labels, "station")  # <-- Match name

    df_encoded.printSchema()

    # No need to drop again here (already dropped inside expand_onehot)
    # But if you keep it, make sure names match:
    # df_encoded = df_encoded.drop("hour_vec", "weekday_vec", "station_vec")  # <-- Fix column name here

    print("✅ Finished one-hot encoding with readable column names")
    return df_encoded

    

def expand_onehot(df, vec_col, labels, prefix):
    """
    Turns a one-hot vector column into individual columns with label-based names.
    """
    df = df.withColumn(f"{vec_col}_arr", vector_to_array(vec_col))
    for i, label in enumerate(labels):
        clean_label = re.sub(r"\W+", "", label)  # Remove special chars just in case
        df = df.withColumn(f"{prefix}_{clean_label}", col(f"{vec_col}_arr")[i])
        print(f"Creating column {prefix}_{clean_label}")
    return df.drop(vec_col).drop(f"{vec_col}_arr")


def save_raw_data(df):
    df.write.mode("overwrite").option("header", True).csv("./final_output/raw_data")



def main():
    weather_df = load_weather_data()
    weather_df = broadcast(weather_df)
    process_bike_files(weather_df)
    batch_paths = sorted(str(p) for p in Path("./processed_dataset").glob("batch_*"))

    # Read and union in batches
    combined_df = None

    for path in batch_paths:
        df = spark.read.option("header", True).csv(path)
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.unionByName(df)

    # Optional: coalesce to reduce output file count
    combined_df = combined_df.coalesce(1)

    # Write to final output directory
    combined_df.write.mode("overwrite").option("header", True).csv("./intermediate")

    final_df = spark.read.csv("./intermediate", header=True)
    save_processed_data(final_df)
    save_raw_data(final_df)
    if os.path.exists("./processed_dataset"):
        shutil.rmtree("./processed_dataset")
    if os.path.exists("./intermediate"):
        shutil.rmtree("./intermediate")
    
    

main()

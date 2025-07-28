
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, date_trunc, col, when, avg, to_timestamp, regexp_replace, broadcast, hour, date_format
from pyspark.sql.types import IntegerType, DoubleType
import shutil
from pathlib import Path
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler, VectorAssembler, StringIndexer
import os
import subprocess
import zipfile

spark = SparkSession.builder.appName("myApp").getOrCreate()

bike_folder_path = "./historical_datasets/Bike Historical Data"

weather_path = "./historical_datasets/Weather Historical Data/weather_data.csv"

DEFAULT_NULL_VALUE = "0"
NULL_VALUE_PATTERN = "\\N"
ZERO_CAPACITY = 0
MAX_AVAILABILITY = 1.0
AVAILABILITY_CAP_THRESHOLD = 1.0
FINAL_PARTITION_COUNT = 1



## MAIN METHOD OF SCRIPT ##
def main():
    generate_bike_files()
    weather_df = load_weather_data()
    process_bike_files(weather_df)
    final_df = combine_data()
    save_processed_data(final_df)
    save_raw_data(final_df)
    remove_intermediate_files()

## GENERATES HISTORICAL BIKE DATA ##
def generate_bike_files():
    bash_command = """
    curl -L -o ~/Downloads/citi-bike-stations.zip \\
    https://www.kaggle.com/api/v1/datasets/download/rosenthal/citi-bike-stations
    """
    subprocess.run(bash_command, shell=True, check=True)

    zip_path = os.path.expanduser("~/Downloads/citi-bike-stations.zip")
    extract_to = "./historical_datasets/Bike Historical Data"
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)





## FEATURE ENGINEERING FOR WEATHER DATA ##
def load_weather_data():
    weather_df = spark.read.csv(weather_path, header=True)
    weather_df = clean_weather_table(weather_df)
    weather_df = broadcast(weather_df)
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
    return df.groupBy("time").agg(avg("HourlyDryBulbTemperature"))





## FEATURE ENGINEERING FOR BIKE DATA ##
def process_bike_files(weather_df):
    print("check for process_bike_files")
    bike_file_list = retrieve_bike_file_lists()

    for i, file in enumerate(bike_file_list):
        print(f"loading {file} into Spark")
        bike_df = load_bike_data(file)
        joined_data = bike_df.join(weather_df, on=["time"])
        joined_data.printSchema()
        joined_data.show(1, truncate=False)
        joined_data.write.mode("overwrite").option("header", True).parquet(f"./processed_dataset/batch_{i}")

def retrieve_bike_file_lists():
    return [os.path.join(bike_folder_path, f) for f in os.listdir(bike_folder_path) if f.endswith(".csv")]

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
    return (when((col(column_name)) == NULL_VALUE_PATTERN, DEFAULT_NULL_VALUE).otherwise(col(column_name))).cast(IntegerType())

def clean_bike_table(df):
    df = df.withColumn("availability", calculate_bike_availability(df))
    df = truncate_time_bike(df)
    df = df.select("time","station_id", "availability")
    return df.groupBy("time", "station_id").agg(avg("availability"))

def calculate_bike_availability(df):
    availability = (df.num_bikes_available + df.num_ebikes_available) / df.capacity
    return when(df.capacity == ZERO_CAPACITY, MAX_AVAILABILITY).when(availability > MAX_AVAILABILITY, AVAILABILITY_CAP_THRESHOLD).otherwise(availability)

def truncate_time_bike(df):
    df = df.withColumn("date", from_unixtime("station_status_last_reported"))
    df = df.withColumn("time", date_trunc("hour", "date"))
    return df

## COMBINES DATA FROM ALL BATCHES ##
def combine_data():
    batch_paths = sorted(str(p) for p in Path("./processed_dataset").glob("batch_*"))

    combined_df = None

    for path in batch_paths:
        df = spark.read.option("header", True).parquet(path)
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.unionByName(df)

    combined_df = combined_df.coalesce(1)
    combined_df.write.mode("overwrite").option("header", True).parquet("./intermediate")
    final_df = spark.read.parquet("./intermediate", header=True)
    return final_df


## ENCODES AND SAVES DATA AS SPARK DFs ##
def save_processed_data(df):
    df = df.withColumn("hour", hour("time"))
    df = df.withColumn("weekday", date_format("time", "EEEE"))
    df = normalize_temperature(df)
    df = one_hot_encode_data(df)
    df.printSchema()
    df = df.withColumn("avg(availability)", col("avg(availability)").cast(DoubleType()))
    df = df.drop(
        "time",
        "station_id",
        "hour",
        "weekday",
        "avg(HourlyDryBulbTemperature)",
        "temp_vec",
        "hour_index",
        "weekday_index",
        "station_id_index"
    )
    df.write.mode("overwrite").parquet("./final_output/cleaned_data")

def normalize_temperature(df):
    df = df.withColumn("avg(HourlyDryBulbTemperature)", col("avg(HourlyDryBulbTemperature)").cast(DoubleType()))
    temp_assembler = VectorAssembler(inputCols=["avg(HourlyDryBulbTemperature)"], outputCol="temp_vec")
    vector_dataframe = temp_assembler.transform(df)

    # Apply MinMaxScaler
    scaler = MinMaxScaler(inputCol="temp_vec", outputCol="temperature_scaled")
    scaler_model = scaler.fit(vector_dataframe)

    # Step 2: Transform to get scaled vector column
    df_scaled = scaler_model.transform(vector_dataframe)

    print("finished normalizing temperature")
    return df_scaled

def one_hot_encode_data(df):
    hour_indexer = StringIndexer(inputCol="hour", outputCol="hour_index").fit(df)
    weekday_indexer = StringIndexer(inputCol="weekday", outputCol="weekday_index").fit(df)
    station_indexer = StringIndexer(inputCol="station_id", outputCol="station_id_index").fit(df)

    df = hour_indexer.transform(df)
    df = weekday_indexer.transform(df)
    df = station_indexer.transform(df)

    encoder = OneHotEncoder(
        inputCols=["hour_index", "weekday_index", "station_id_index"],
        outputCols=["hour_vec", "weekday_vec", "station_vec"],
        dropLast= False  
    )
    df_encoded = encoder.fit(df).transform(df)

    df_encoded.printSchema()

    print("✅ Finished one-hot encoding with readable column names")
    return df_encoded

def save_raw_data(df):
    df.write.mode("overwrite").option("header", True).parquet("./final_output/raw_data")


## REMOVES INTERMEDIATE FILES ##
def remove_intermediate_files():
    if os.path.exists("./intermediate"):
        shutil.rmtree("./intermediate")
    if os.path.exists("./processed_dataset"):
        shutil.rmtree("./processed_dataset")

main()

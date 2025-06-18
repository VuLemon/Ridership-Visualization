
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, date_trunc, col, when, avg, to_timestamp, regexp_replace, broadcast, hour, date_format
from pyspark.sql.types import IntegerType
import shutil
import pandas as pd
import sklearn

import os

spark = SparkSession.builder.appName("myApp").getOrCreate()

bike_folder_path = "./historical_datasets/Bike Historical Data"

weather_path = "./historical_datasets/Weather Historical Data/weather_data.csv"

bike_file_list = [os.path.join(bike_folder_path, f) for f in os.listdir(bike_folder_path) if f.endswith(".csv")] # Makes sure it's a CSV file

def main():
    weather_df = load_weather_data()
    weather_df = broadcast(weather_df)
    process_bike_files(weather_df)
    final_df = spark.read.csv("./processed_dataset/*", header= True).toPandas()
    save_one_hot_encoded_data(final_df)
    save_raw_data(final_df)
    if os.path.exists("./processed_dataset"):
        shutil.rmtree("./processed_dataset")


    

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
    df = df.withColumn("hour", hour("time"))
    df = df.withColumn("weekday", date_format("time", "EEEE"))
    df = df.select("weekday", "hour", "HourlyDryBulbTemperature")
    return df

def cast_temperature_as_integer(df):
    df = df.withColumn("HourlyDryBulbTemperature", regexp_replace("HourlyDryBulbTemperature",'s$',''))
    df = df.withColumn("HourlyDryBulbTemperature", col("HourlyDryBulbTemperature").cast(IntegerType()))
    return df.groupBy("hour","weekday").agg(avg("HourlyDryBulbTemperature"))




def process_bike_files(weather_df):
    for i, file in enumerate(bike_file_list):
        bike_df = load_bike_data(file)
        joined_data = bike_df.join(weather_df, on=["weekday", "hour"])
        joined_data.printSchema()
        joined_data.show(1, truncate=False)
        joined_data.write.mode("overwrite").option("header", True).csv(f"./processed_dataset/batch_{i}")

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
    df = df.select("station_id", "availability", "hour", "weekday")
    return df.groupBy("hour", "weekday", "station_id").agg(avg("availability"))

def calculate_bike_availability(df):
    availability = (df.num_bikes_available + df.num_ebikes_available) / df.capacity
    return when(df.capacity == 0, 1).when(availability > 1, 1).otherwise(availability)

def truncate_time_bike(df):
    df = df.withColumn("date", from_unixtime("station_status_last_reported"))
    df = df.withColumn("time", date_trunc("hour", "date"))
    df = df.withColumn("hour", hour("time"))
    df = df.withColumn("weekday", date_format("time", "EEEE"))
    return df
    # return availability





def save_one_hot_encoded_data(df):
    enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output= False)
    df.head(5)
    one_hot_encoded = enc.fit_transform(df[['weekday', 'hour', 'station_id']])
    enc.categories_

    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns = enc.get_feature_names_out(['weekday','hour','station_id'])
    )

    df_encoded = pd.concat([df, one_hot_df], axis=1)

    df_encoded = df_encoded.drop(['weekday','hour','station_id'], axis=1)

    df_encoded.to_csv("final_output/clean_data.csv", index=False, header=True)

def save_raw_data(df):
    df.to_csv("final_output/raw_data.csv", index=False, header=True)


main()



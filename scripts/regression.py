from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()

results = []

def main():
    df = spark.read.parquet("./final_output/cleaned_data")
    final_data = transform_features_into_vector(df)
    train_data, test_data = generate_historical_and_test(final_data)
    generate_linear_regressions(train_data, test_data)
    generate_forest_regressions(train_data, test_data)
    generate_GBT_regressions(train_data, test_data)

def transform_features_into_vector(df):
    assembler = VectorAssembler(
        inputCols=["temperature_scaled", "hour_vec", "weekday_vec", "station_vec"],
        outputCol="features"
    )

    df_vectorized = assembler.transform(df)
    final_data = df_vectorized.select("features", "avg(availability)")
    return final_data

def generate_historical_and_test(final_data):
    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
    return train_data, test_data

def generate_linear_regressions(train_data, test_data):

    lr = LinearRegression(featuresCol="features", labelCol="avg(availability)")
    lr_model = lr.fit(train_data)
    rmse, r2 = evaluate_model(lr_model, test_data)
    results.append(("Linear Regression", rmse, r2))

    ridge = LinearRegression(featuresCol="features", labelCol="avg(availability)", regParam=0.1, elasticNetParam=0)
    ridge_model = ridge.fit(train_data)
    rmse, r2 = evaluate_model(ridge_model, test_data)
    results.append(("Linear Regression - Ridge", rmse, r2))

    lasso = LinearRegression(featuresCol="features", labelCol="avg(availability)", regParam=0.1, elasticNetParam=1)
    lasso_model = lasso.fit(train_data)
    rmse, r2 = evaluate_model(lasso_model, test_data)
    results.append(("Linear Regression - Lasso", rmse, r2))


def generate_forest_regressions(train_data, test_data):
    rf = RandomForestRegressor(featuresCol="features", labelCol="avg(availability)", numTrees=10)
    rf_model = rf.fit(train_data)
    rmse, r2 = evaluate_model(rf_model, test_data)
    results.append(("Random Forest", rmse, r2))



def generate_GBT_regressions(train_data, test_data):
    gbt = GBTRegressor(featuresCol="features", labelCol="avg(availability)", maxIter=10)
    gbt_model = gbt.fit(train_data)
    rmse, r2 = evaluate_model(gbt_model, test_data)
    results.append(("GBT Regressor", rmse, r2))


def evaluate_model(model, test_data, label="avg(availability)"):
    predictions = model.transform(test_data)
    evaluator_rmse = RegressionEvaluator(
        labelCol=label, predictionCol="prediction", metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=label, predictionCol="prediction", metricName="r2"
    )
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    return rmse, r2

main()
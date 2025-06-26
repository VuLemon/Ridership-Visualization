**CITI-BIKE AVAILABILITY PREDICTION REPORT**

1. **Introduction**:
This project aims to generate a prediction of the availability of citi-bikes in New York, based on 4 factors listed below:
- Station location
- Time of day 
- Day of week
- Temperature

The project generates these predictions based on historical data, using weather reports from NOAA (LaGuardia Station), and station status from citi-bikes own public APIs. The historical data is set during 2019-2021.



For this project, I want to know if factors such as time, location and weather conditions play a major role in people's decision to use bikes. While it may sound intuitive (obviously weather condition will affect the ridership), I want to be able to back it up with data, and more importantly, identify how much of an influence these factors can have on the ridership.

2. **Methodology**

For this prediction model, we have 4 features, and 1 target label: 
Availability: This is our target label, which we will predict based on the other 4 features
Station location: For the purpose of this project, we will be using station IDs, which corresponds to citi-bike stations locations across New York. 
Time of day + day of week: These can be generated from the timestamp in our dataset
Temperature: We will be using Hourly Dry Bulb Temperature from the weather dataset included

    a. Data collection and cleaning:
The bike dataset comes from here: https://www.kaggle.com/datasets/rosenthal/citi-bike-stations/data
The weather dataset comes from here: https://www.ncei.noaa.gov/cdo-web/datatools/lcd

I collected station data and weather data over the period of 2019 - 2021. For the bike dataset, some of the entries are missing certain important features, such as station availability, which means I have to drop them. The dataset is quite large, and spread over 50 .csv files, totalling about 26gb of data. While it's good to have such an extensive historical record, this presents its own challenges in implementation. 

I then join each of the citi-bike file with the weather file to produce the final dataset

    b. Feature Engineering:
Certain features are derived from the dataset for this project:
- Availability: I calculate the availability of a station through this formula: (number_of_bike + number_of_e-bike)/station_availability. This gives me an estimate of how available a station is, and how often people are using bikes from said station
- Time of day + day of week: I extract this data from the timestamp. My assumption is that certain days will be more crowded than others (think weekdays vs weekends), and that certain timeblocks will see more ridership (6AM compared to 3AM for example). As such, I treated these features as categorical, and one hot encode them for my regression models.
- Station location: This feature is included in the dataset, and can be used as is. As with categorical features, I one-hot encode these as well
- Temperature: This feature is included in the dataset, and can be used as is. For this feature, I normalize the values.

These features are then fed into different regression models, so that I can compare their performance.

    c. Modelling:
For this project, I decided to use these regression models:
- Linear Regression

    - Ridge Regression

    - Lasso Regression

- Random Forest Regressor

- Gradient Boosted Trees (GBT)

I want to compare the performance of multiple models, based on their RMSE (Root Mean Squared Error) and R² Score (Coefficient of Determination). These are the results:

| Model             | RMSE   | R² Score |
| ----------------- | ------ | -------- |
| Linear Regression | 0.2722 | 0.2935   |
| Random Forest     | 0.3210 | 0.0179   |
| GBT Regressor     | 0.3180 | 0.0361   |
| Ridge Regression  | 0.2753 | 0.2774   |
| Lasso Regression  | 0.3239 | -0.0000  |


 Based on this score, it can be seen that Linear Regression performs the best, with an RMSE score of ~0.27, and an R² score of ~0.29. The performance is quite modest, with the best model only explaining around ~29% of the variance. 

3. **Challenges Faced**
One of the biggest challenge I faced when working on this project is the size of the bike dataset (26gb), which makes loading the historical data for preprocessing a big challenge. Loading all 50 bike .csv files and joining with the weather .csv file would frequently cause a memory error and abort the Spark job. As a result, I split the joining into batches, where I would join each small file with the weather data and write the intermediate result into files. While this increases the number of operations needed to be performed, my device's local memory is just enough that it works out.

The final data also has to be saved as parquet files, as they are the only file types that fit within my device's memory. Loading into Pandas dataframe is also not an option, and as such I was limited to PySpark's own ML library, rather than using other libraries such as sci-kit learn. However, I was really happy with intuitive the library was. 


4. **Future Plans**
Linear Regression performed best among the models tested, though overall R² values suggest room for improvement. In future iterations, I would group certain features together (morning, noon, evening and night instead of actual time, for exmaple), to avoid overfitting. Despite the modest performance, this project helped solidify my understanding of regression modeling and PySpark workflows at scale.
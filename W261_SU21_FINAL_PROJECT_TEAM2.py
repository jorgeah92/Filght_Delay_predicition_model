# Databricks notebook source
# MAGIC %md #Flight Delay Prediction Model
# MAGIC 
# MAGIC ## w261 - Machine Learning at Scale, Summer 2021
# MAGIC #### Team 2 - Bo Qian, Demian Rivera, Devin Robison, Jorge Hernandez

# COMMAND ----------

# MAGIC %md ## 1. Question Formulation
# MAGIC 
# MAGIC Given the complex logistics and congestion in the airline industry as a result of weather and non-weather related events and its impact on travel profitability, it has become increasingly important to more accurately predict flight delays in order to increase customer satisfaction, and thereafter customer loyalty. Especially in an environment where social media can act as a catalyst for multiplying the negative effect of a bad customer experience. Given our confidence in this relationship between customer experience and profitability, our stakeholders include customers, consumers, airline companies, and stockholders.
# MAGIC 
# MAGIC It’s important to note that we believe that customers care more about arrival delay than departure delay and given that a significant amount of delayed time is made up by pilots in the air, we have decided to focus on predicting flight arrival time delay, which we believe is more associated with a negative customer experience.
# MAGIC 
# MAGIC We explore and contrast four different models, Logistic Regression, Decision Tree, Random Forest, and XG Boost based off of Precision and Recall. We decided to focus on these metrics as they are the most representative of relevance. In order for any model to be of any practical use, we have decided to compare these models and also set a Precision threshold at 80% and Recall at 70%. These thresholds ensure that we reliably predict what customers and other stakeholders care about most.
# MAGIC 
# MAGIC **Literature review:**
# MAGIC 
# MAGIC Predicting flight delays has attracted many experts and driven many studies. Overall, the predictive methods have been grouped into  five groups, Statistical methods, Probability methods, Network-based methods, Operational Methods, and Machine Learning Methods. Due to the complexity and volatility of parameters and cross effect on eachother, the problem is considered to be NP Complete and or non-linear nature. In addition, it’s important to consider that location is of high significance. For example, weather conditions are associated with ~70% of delays in the USA but under 4% in Europe, so including weather data might be more relevant in the latter. 
# MAGIC Today, intelligent and deep neural nets are used in actual applications in controlling traffic progress but LightGBM, Multilayer Perceptron, and Random Forest is also widely used for this field of study. Unfortunately, there is no perfect one-size-fits-all and different models have proved to be best for different problems [1]. One of the most interesting studies, albright somewhat dates now, [2] used neural nets and Deep Believe Nets (based on Boltzmann machine) to train model. The model uses Momento and Resilient Back Optimization Propagation.
# MAGIC More in line with our study,  and with great results, a group or researchers [3] analized a similar dataset than ours but focused on 5 airports in the US. Due to imbalanced data, Randomized SMOTE was applied for Data Balancing, They used a Gradient Boosting Classifier Model as well as Grid Search for hyper-parameter tuning achieving a max accuracy of 85.73%
# MAGIC 
# MAGIC 1 Yazdi, M.F., Kamel, S.R., Chabok, S.J.M. et al. Flight delay prediction based on deep learning and Levenberg-Marquart algorithm. J Big Data 7, 106 (2020).
# MAGIC 
# MAGIC 2 Venkatesh V, et al. Iterative machine and deep learning approach for aviation delay prediction. In 2017 4th IEEE Uttar Pradesh Section International Conference on Electrical, Computer and Electronics (UPCON). New York: IEEE; 2017.
# MAGIC 
# MAGIC 3 Chakrabarty N. A data mining approach to flight arrival delay prediction for american airlines. In 2019 9th Annual Information Technology, Electromechanical Engineering and Microelectronics Conference (IEMECON). New York: IEEE; 2019.

# COMMAND ----------

# MAGIC %md ## 2. Exploratory Data Analysis
# MAGIC 
# MAGIC Useful Resources:
# MAGIC + Airlines data fields description: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
# MAGIC + Airline delay causes: https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp
# MAGIC + Why flight delays: https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC + Weather data fields description: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf >> from page 5

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import functions as f
from pyspark.sql.functions import udf

from pyspark.sql.window import Window

from pyspark.sql.types import  IntegerType
from pyspark.sql.types import DoubleType

from IPython.display import HTML

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

from functools import partial

# To help build the models
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer

from pyspark.ml import Pipeline

# Classification models
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from sparkdl.xgboost import XgboostClassifier

# Evaluators
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# MAGIC %md #####2.1 General EDA

# COMMAND ----------

# MAGIC %md Below we are setting up a "local" storage so that we can save our clean and merged dataset to save time in composing when running the code after starting up a cluster.

# COMMAND ----------

blob_container = "261-final-project" # The name of your container created in https://portal.azure.com
storage_account = "mids261storage" # The name of your Storage account created in https://portal.azure.com
secret_scope = "mids261" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "mids261key2" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# Read data: Airlines, Weather, Global Weather Stations, Global Airports
df_airports = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(f"{blob_url}/GlobalAirportDatabase.csv")
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/stations_with_neighbors.parquet")
airlines = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/201*.parquet/")
airlines_6m = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data_6m/*")
weather = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather201*.parquet/")

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

# MAGIC %md We can see that the `airline` dataset contains many columns. Some of these columns were not helpful in our model and required further investiagate to determine its usefulness and any further actions that were required to prepare the data for running in the models.

# COMMAND ----------

airlines.describe()

# COMMAND ----------

display(airlines)

# COMMAND ----------

weather.printSchema()

# COMMAND ----------

# MAGIC %md The `weather` dataset fell into the same position as the `airline` dataset. Most of the columns were not useful for our model. Those that were useful needed further action to prepare for use in our model since they are in a csv string format but with numerical values.

# COMMAND ----------

display(weather)

# COMMAND ----------

# convert the spark df into a pandas df for easy plotting:
airlines_pandas_df = airlines_6m.toPandas()

# COMMAND ----------

# MAGIC %md The smaller airline dataset is used for visualizations since it is able to be converted into a pandas dataframe and therefore easier to plot.

# COMMAND ----------

# plot the distribution of each feature:
airlines_pandas_df.hist(figsize=(30,30), bins=50)
display(plt.show())

# COMMAND ----------

# plot the correlation matrix of the features:
df = airlines_6m_pandas_df.corr()
HTML(df.to_html(classes='table table-striped'))

# COMMAND ----------

# MAGIC %md The correlation chart shows that there are a large number of features which are either redundant for our analysis, such as ‘DIV[N]_XXX’, or those which are implicitly correlated with our dependent variable and can be aggregated, such as ‘ARR_DEL15’, ‘CANCELLED’, ‘DIVERTED’, ‘CANCELLATION CODE’ etc... Given that our prediction target is whether or not a plane with be delayed, our baseline features should incorporate all information available up to the point of initially predicted departure such as ‘ORIGIN’, ‘DISTANCE’, ‘OP_UNIQUE_CARRIER’. 

# COMMAND ----------

#Summary stats for departure and arrival delays

print(airlines_6m_pandas_df[['DEP_DELAY', 'ARR_DELAY']].agg([np.mean, np.median, np.max, np.min]))

# COMMAND ----------

# What is the Carrier distribution?

print(airlines_6m_pandas_df['OP_UNIQUE_CARRIER'].value_counts())
print(airlines_6m_pandas_df['OP_UNIQUE_CARRIER'].value_counts(normalize = True)*100)  

# COMMAND ----------

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('OP_UNIQUE_CARRIER','CARRIER_DELAY', data=airlines_6m_pandas_df,ax=ax[0])
ax[0].set_title('Average Delay by Carrier')
sns.boxplot('OP_UNIQUE_CARRIER','CARRIER_DELAY', data=airlines_6m_pandas_df,ax=ax[1])
ax[1].set_title('Delay Distribution by Carrier')
plt.close(2)
plt.show()

# COMMAND ----------

# MAGIC %md The above graphs indicates that there are specific carriers that experience more delays than others. Therefore each carries should be represented in the model.

# COMMAND ----------

# Display histogram of departure delays in minutes
plt.rcParams["figure.figsize"] = (15,4)
bins, counts = airlines.filter("DEP_DELAY  IS NOT NULL") \
                            .filter('CANCELLED == False') \
                            .filter('DIVERTED == False') \
                            .filter("DEP_DELAY  < 120") \
                            .filter("DEP_DELAY  > 0") \
                            .selectExpr("DEP_DELAY").rdd.flatMap(lambda x: x).histogram(30)
plt.hist(bins[:-1], bins=bins, weights=counts)
plt.xlabel('Departure Delay (minutes)')

# COMMAND ----------

# Display histogram of arrival delays in minutes
plt.rcParams["figure.figsize"] = (15,4)
bins, counts = airlines.filter("ARR_DELAY  IS NOT NULL") \
                            .filter('CANCELLED == False') \
                            .filter('DIVERTED == False') \
                            .filter("ARR_DELAY  < 120") \
                            .filter("ARR_DELAY  > 0") \
                            .selectExpr("ARR_DELAY").rdd.flatMap(lambda x: x).histogram(30)
plt.hist(bins[:-1], bins=bins,weights=counts)
plt.axvline(x=15, color='red')
plt.xlabel('Arrival Delay (minutes)')

# COMMAND ----------

# EDA on binary outcome variable
airlines_eda_arr = airlines.withColumn('ARRIVAL_STATUS', f.when((f.col("ARR_DELAY") > 15) | \
                                          (f.col("CANCELLED") == "true") | \
                                          (f.col("DIVERTED") == "true"), 'delayed')\
                                          .otherwise('on time'))
airlines_eda_arr.createOrReplaceTempView("airlines_eda_arr")
display(spark.sql("select ARRIVAL_STATUS,count(*) as count from airlines_eda_arr group by ARRIVAL_STATUS order by count desc"))

# COMMAND ----------

# MAGIC %md When looking at the chosen label, ARR_DELAY, it can be seen that there are a larger number of examples of instances with no delays than there are of instanes of delays in flight. This inbalance should be expected in this context since we would expect more flights to be on time then delayed or the flight system would be in constant chaos. Any flights over 15 minutes late will be considered as delayed; this is the standard in the field. 

# COMMAND ----------

# MAGIC %md #####2.2 Rush Month & Intraday Rush Hour Blocks
# MAGIC 
# MAGIC After graphing the number of departure and arrival delays on blocks of time throughout the day or by month, we noticed that there are an increase in the number of delays during certain times. June, July and August have more departure and arrival delays than other months potentially because of the higher travel volume for summer vacations. An interesting finding is departure delays and arrival delays have different intraday rush hours: the rush hour for departure delays is 15~20pm, whereas 16~22pm is the rush block for arrival delays. 

# COMMAND ----------

#number of departure delays by month
departure_delay_byMonth = spark.sql("select count(DEP_DELAY) as DELAY_COUNTS, MONTH from airlines where DEP_DELAY > 15 group by MONTH order by MONTH").toPandas()
ax2 = arrive_delay_byMonth.boxplot(column=['DELAY_COUNTS'], by=['MONTH'])
ax2.set_xlabel('Month'); ax1.set_ylabel('Number of Delays')
ax2.get_figure().suptitle('')
ax2.get_figure().gca().set_title("")
ax2.set_title('Number of Depature Delays by Month')
plt.show()

# COMMAND ----------

#number of arrival delays by month
arrive_delay_byMonth = spark.sql("select count(ARR_DELAY) as DELAY_COUNTS, MONTH from airlines where ARR_DELAY > 15 group by MONTH order by MONTH").toPandas()
ax2 = arrive_delay_byMonth.boxplot(column=['DELAY_COUNTS'], by=['MONTH'])
ax2.set_xlabel('Month'); ax1.set_ylabel('Number of Delays')
ax2.get_figure().suptitle('')
ax2.get_figure().gca().set_title("")
ax2.set_title('Number of Arrival Delays by Month')
plt.show()

# COMMAND ----------

#number of departure delays by hour block
plt.rcParams["figure.figsize"] = (25,6)
departure_delay_byHour = spark.sql("select count(DEP_DELAY) as DELAY_COUNTS, DEP_TIME_BLK from airlines where DEP_DELAY > 15 group by DEP_TIME_BLK order by DEP_TIME_BLK").toPandas()
ax3 = departure_delay_byHour.boxplot(column=['DELAY_COUNTS'], by=['DEP_TIME_BLK'])
ax3.set_xlabel('Hour Block'); ax1.set_ylabel('Number of Delays')
ax3.get_figure().suptitle('')
ax3.get_figure().gca().set_title("")
ax3.set_title('Number of Departure Delays by Intraday Hour Block')
plt.show()

# COMMAND ----------

#number of arrival delays by hour block
airlines.createOrReplaceTempView("airlines")
plt.rcParams["figure.figsize"] = (25,6)
arrive_delay_byHour = spark.sql("select count(ARR_DELAY) as DELAY_COUNTS, ARR_TIME_BLK from airlines where ARR_DELAY > 15 group by ARR_TIME_BLK order by ARR_TIME_BLK").toPandas()
ax1 = arrive_delay_byHour.boxplot(column=['DELAY_COUNTS'], by=['ARR_TIME_BLK'])
ax1.set_xlabel('Hour Block'); ax1.set_ylabel('Number of Delays')
ax1.get_figure().suptitle('')
ax1.get_figure().gca().set_title("")
ax1.set_title('Number of Arrival Delays by Intraday Hour Block')
plt.show()

# COMMAND ----------

# MAGIC %md ## 3. Feature Engineering

# COMMAND ----------

# MAGIC %md #####3.1 Preprocess Airlines Data
# MAGIC 
# MAGIC Based on the findings and correlations from the EDA, we decided to remove columns that are not highly related to delay variables to reduce the data size. At this stage, we only kept below variables:
# MAGIC * flight descriptive features: YEAR, MONTH, DAY_OF_WEEK, OP_UNIQUE_CARRIER, ORIGIN, DEST
# MAGIC * DEP and ARR features: 'CRS_DEP_TIME','DEP_TIME', 'DEP_DELAY','DEP_TIME_BLK','DEP_DEL15'; 'CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_TIME_BLK','ARR_DEL15'
# MAGIC * delay reasons: 'CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'
# MAGIC 
# MAGIC We also filtered out cancelled and diverted flights. Based on intuition, passengers will usually be pre-notified about flight cancellations or diversions even before their check-in, so our predictions will mainly focus on those “real” delays. 
# MAGIC 
# MAGIC Additionally, we introduced three new binary variables based on the findings in the EDA about month seasonality and intraday rush hour blocks as below:
# MAGIC * IS_RushMonth: whether the flight date is in June, July or August. 
# MAGIC * IS_DEP_RUSH_HOUR: whether the departure hour is between 15:00 to 20:00
# MAGIC * IS_ARR_RUSH_HOUR: whether the arrival hour is between 16:00 to 22:00

# COMMAND ----------

def is_RushMonth(x):
  """
  Function to determine if a flight is in a rush month (June,July,August)
  """
  if   x in [6,7,8]: 
    return 1
  else: 
    return 0


def is_DEP_RushHour(x):
  """
  Function to determine if a departure time falls into rush hour (1500-2000)
  """
  if (x != None) and (x >= 1500) and (x <= 2000): 
    return 1
  else: 
    return 0
  
def is_ARR_RushHour(x):
  """
  Function to determine if an arrival time falls into rush hour (1600-2100)
  """
  if (x != None) and (x >= 1600) and (x <= 2200): 
    return 1
  else: 
    return 0

# function to preprocess Airlines:
def preprocessAirlines(df):
  cols_to_keep = ['YEAR','MONTH', 'IS_RushMonth', 'DAY_OF_WEEK', 'FL_DATE','OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
                  'CRS_DEP_TIME','DEP_TIME','IS_DEP_RUSH_HOUR','DEP_DELAY','DEP_TIME_BLK','DEP_DEL15',
                  'CRS_ARR_TIME','ARR_TIME','IS_ARR_RUSH_HOUR','ARR_DELAY','ARR_TIME_BLK','ARR_DEL15',
                  'CRS_ELAPSED_TIME','DISTANCE',
                  'CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
  cols_to_remove = [x for x in df.columns if x not in cols_to_keep]
  df = df.orderBy("FL_DATE") 
  df = df.filter(df.CANCELLED == False)
  df = df.filter(df.DIVERTED == False)
  df = df.withColumn('CARRIER_DELAY', f.when(df.CARRIER_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('WEATHER_DELAY', f.when(df.WEATHER_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('NAS_DELAY', f.when(df.NAS_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('SECURITY_DELAY', f.when(df.SECURITY_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn('LATE_AIRCRAFT_DELAY', f.when(df.LATE_AIRCRAFT_DELAY.isNotNull(), 1).otherwise(0))
  df = df.withColumn("IS_RushMonth", f.udf(is_RushMonth, IntegerType())("MONTH"))
  df = df.withColumn("IS_DEP_RUSH_HOUR", f.udf(is_DEP_RushHour, IntegerType())("DEP_TIME"))
  df = df.withColumn("IS_ARR_RUSH_HOUR", f.udf(is_ARR_RushHour, IntegerType())("ARR_TIME"))
  df = df.fillna(0, subset=['ARR_DELAY', 'DEP_DELAY'])
  df = df.withColumn('DEP_DEL15', f.when(df.DEP_DELAY > '15', 1).otherwise(0))
  df = df.withColumn('ARR_DEL15', f.when(df.ARR_DELAY > '15', 1).otherwise(0))
  preprocessAirlines_df = df.drop(*cols_to_remove)
  return preprocessAirlines_df

airlines_preprocessed = preprocessAirlines(airlines)
display(airlines_preprocessed.head(5))

# COMMAND ----------

# MAGIC %md #####3.2 Preprocess Weather Data
# MAGIC 
# MAGIC To reduce the data size, we first filtered out all the non US weather observations based on their longitude and latitude.
# MAGIC 
# MAGIC The Weather dataset contains several combined columns. For example, the WIND column contains four parts of information, as well as column CIG, VIS etc. We parsed every combined column into separate columns. After parsing, erroneous and missing data represented by 999, 9999 or 99999 were filtered out. 
# MAGIC 
# MAGIC Lastly, only below features were kept as they have less missing values and we considered them as the main weather data features:
# MAGIC * observation basic information: STATION, DATE, ELEVATION, LATITUDE, LONGTITUDE
# MAGIC * wind features: WND_ANGLE, WND_SPEED
# MAGIC * other weather features: CIG_HIGHT, VIS_DIST, TMP_F, DEW_0, SLP_0

# COMMAND ----------

# functions to preprocess Weather data

def US_fn(df):
  """
  Reduce df to US only
  """
  # US is lat/long ranges according to format: [[(lat_low, lat_high),(long_low, long_high)], [(lat_low, lat_high),(long_low, long_high)]]
  US = [[(24,49),(-125,-67)],[(17,19),(-68,-65.5)], [(13,14),(144,145)], [(15,16),(145,146)], [(-15,-14), (-171,-170)], [(18,19),(-65.4,-64)], [(18,23),(-160,-154)], [(50,175),(-170,-103)]]  

  list_df = [] #empty list for parquet parts
  parquet_part = spark.range(0).drop("id") #empty spark df

  #Filtering for individual areas in US
  for item in US:
    parquet_part = df.filter((f.col('Latitude') > item[0][0]) & (f.col('Latitude') < item[0][1]) & (f.col('Longitude') > item[1][0]) & (f.col('Longitude') < item[1][1]))
    list_df.append(parquet_part)

  #Appending each individual US area
  us_df = functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), list_df)

  return us_df


def preprocessWeather(weather):
        """
        Parse combined columns into separate columns. Filter out low quality data.
        """
        WEATHER_COLUMNS = ['STATION', 'DATE','ELEVATION', 'LATITUDE', 'LONGITUDE','WND_ANGLE', 'WND_SPEED', 'CIG_HEIGHT', 'VIS_DIST', 'TMP_F', 'DEW_0', 'SLP_0']
    
        split_weather_field = f.split(weather['WND'], ',')
        weather = weather.withColumn('WND_ANGLE', split_weather_field.getItem(0).cast("double")) #!=999
        weather = weather.withColumn('WND_SPEED', split_weather_field.getItem(3).cast("double")) #!=9999

        split_weather_field = f.split(weather['CIG'], ',')
        weather = weather.withColumn('CIG_HEIGHT', split_weather_field.getItem(0).cast("double")) #!=99999

        split_weather_field = f.split(weather['VIS'], ',')
        weather = weather.withColumn('VIS_DIST', split_weather_field.getItem(0).cast("double"))#!=999999

        split_weather_field = f.split(weather['TMP'], ',')
        weather = weather.withColumn('TMP_F', split_weather_field.getItem(0).cast("double")) #!=9999

        split_weather_field = f.split(weather['DEW'], ',')
        weather = weather.withColumn('DEW_0', split_weather_field.getItem(0).cast("double")) #!=9999

        split_weather_field = f.split(weather['SLP'], ',')
        weather = weather.withColumn('SLP_0', split_weather_field.getItem(0).cast("double"))#!=99999

        weather = weather.select(WEATHER_COLUMNS)
        
        fltrs = [
        f.col('WND_ANGLE') != 999,
        f.col('WND_SPEED') != 9999,  
        f.col('CIG_HEIGHT') != 99999,
        f.col('VIS_DIST') != 99999,
        f.col('TMP_F') != 9999,
        f.col('DEW_0') != 9999,
        f.col('SLP_0') != 99999
        ]
    
        for i in fltrs:
          weather = weather.filter(i)
          
        # Seperate DATE Columns to DATE and Hour for airline join
        weather = weather.withColumn("DATE_PART", f.to_date(f.col("DATE")))\
                                 .withColumn("HOUR_PART", f.hour(f.col("DATE"))) 
        
        return weather

weather_us = US_fn(weather)     
weather_preprocessed = preprocessWeather(weather_us)
display(weather_preprocessed.head(5))

# COMMAND ----------

# MAGIC %md #####3.3 Find The Nearest Weather Station for Each Airport
# MAGIC 
# MAGIC To find the weather condition of each airport, we first have to find the nearest weather station for them. So we calculated the Haversine distance between each airport and each weather station, and matched airports with their closed weather stations. 
# MAGIC 
# MAGIC As an initial simplifying assumption we decided to assume a 1 to 1 matching between stations and airports; while this might not be the most robust approach, as some airports have more than one weather station in the same proximity, which could potentially provide a better picture of regional weather patterns, it would have greatly complicated the weather aggregation process which already appeared to be a long running task. Additionally, as we would need to compute distances between each station and each airport in our dataset, and our datasets contain more than N=9300 unique airports and M=2200 weather stations, resulting in more than 20 million calculations and a O(N*M) complexity, it was clear that we would want to prune our data before continuing, even with an embarrassingly parallel task.
# MAGIC 
# MAGIC Since our problem statement contained clear geographic constraints, specifically that we were restricting our predictions to the United States, one immediate optimization was to filter out everything across all tables that fell outside our geographic region of interest. This reduced our total airports to < 2100, and our weather stations to 524, requiring only around 1 million distance calculations or 5% of the original value.
# MAGIC 
# MAGIC After filtering, we computed an intermediate ‘nearest’ table containing airport, station pairs, as well as the latitude and longitude values of each. This process consisted of a left outer join between weather stations and airports, and the parallel computation of Haversine distances between each pair. Subsequently, the dataset was partitioned on airport ID using PySpark window functions, sorted by Haversine distance, and the row with minimal distance is returned.

# COMMAND ----------

## Functions to compute nearest weather station to a given airport
#https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546
lower_48_bb = [-127.23, 24.45, -60.04, 49.46]
def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
def compute_nearest_station(df_airports, df_stations,
                            region_bound={"W": lower_48_bb[0], "S": lower_48_bb[1], "E": lower_48_bb[2], "N": lower_48_bb[3]},
                            airport_dict={"AirportID": "IATA", "AirportLat": "LatDecimalDegree", "AirportLon": "LonDecimalDegree"},
                            station_dict={"StationID": "station_id", "StationLat": "lat", "StationLon": "lon"}
                           ):
  """
    df_airports: Airports dataframe, containing Airport ID, Aiport Latitude, and Airport Longitude fields.
    df_stations: Stations dataframe, containing Station ID, Station Latitude, and Station Longitude.
    region_bound: [West,South,East,North] boundary coordinates to search in. Defaults to lower 48 US states.
    airport_dict: Mapping to identify df fields corresponding to Airport ID/Lat/Lon
    airport_dict: Mapping to identify df fields corresponding to Weather Station ID/Lat/Lon
  """
  _stations = df_stations.select([station_dict["StationID"],
                                  station_dict["StationLat"],
                                  station_dict["StationLon"]])\
                          .distinct()\
                          .toDF("StationID", "StationLat", "StationLon")\
                          .filter(f"StationLat > {region_bound['S']} AND StationLat < {region_bound['N']} AND StationLon > {region_bound['W']} AND StationLon < {region_bound['E']}")
  _airports = df_airports.select([airport_dict["AirportID"],
                                  airport_dict["AirportLat"],
                                  airport_dict["AirportLon"]])\
                            .distinct()\
                            .toDF("AirportID", "AirportLat", "AirportLon")\
                            .filter(f"AirportLat > {region_bound['S']} AND AirportLat < {region_bound['N']} AND AirportLon > {region_bound['W']} AND AirportLon < {region_bound['E']}")
  haversine_udf = udf(haversine)
  _cross_table = _stations.crossJoin(_airports)\
                          .withColumn("Haversine", haversine_udf("StationLat", "StationLon", "AirportLat", "AirportLon"))
  windowSpec = Window.partitionBy("AirportID").orderBy("Haversine")
  df_nearest = _cross_table.withColumn("row_num", row_number()\
                           .over(windowSpec))\
                           .filter("row_num = 1")\
                           .select("AirportID", "StationID", "Haversine", "AirportLat", "AirportLon")
  return df_nearest


df_nearest = compute_nearest_station(df_airports, df_stations)
display(df_nearest)

# COMMAND ----------

# MAGIC %md #####3.4 Merge The Airlines and Weather Datasets
# MAGIC 
# MAGIC Once we know which weather stations are associated with each airport, we’re able to identify all weather events associated with each flight’s departure and arrival airports. But we still need to determine which of those events apply to a given flight. After some discussion, we settled on a time averaging approach that averaged all weather events for a flight within a 12 hour window leading up to its predetermined take off time.
# MAGIC 
# MAGIC The rationale for using an n-hour window centers on the idea that our weather information is already a snapshot of reality, at a point that is nearby, but often not co-located with an airport; as a result, there are few cases where we wouldn’t expect some data smearing to occur. By smearing, we mean that weather information will never translate exactly, it may take hours (if at all) for rain, snow, wind, etc.. observed at a station 40 miles from an airport to manifest, and it may have altered in characteristic by the time it does. Without trying to guess at time delays or construct some complicated model based on how conditions might propagate, we instead opted to average all of the nearest weather events for a time window.
# MAGIC 
# MAGIC To facilitate this we first augmented our flight data so that each flight was assigned a unique FlightID to ensure that our outer joins would not produce incorrect data in the case where two flights had identical information, the appropriate start and end times for weather events it should capture, as well as the departure and arrival weather station identification. Interestingly, working with datetimes in PySpark is not as intuitive as one might expect, as it lacks the ability to directly add hour, minute, and second values to a given datetime field. To get around this, we created a user defined function that splits the ‘DEP_TIME_BLK’ and ‘ARR_TIME_BLK’ fields into hours, constructs a python datetime, and then subtracts a timedelta to set our window bounds.
# MAGIC 
# MAGIC As a final step, another outer join was required, this time between our flight and weather data. This process results in (possibly) multiple rows for each flight, each containing the original flight data, as well as an associated weather event at its departure or arrival airport. Following this, the entire dataset was grouped by flight information, and all associated weather fields were aggregated using a mean value calculation.
# MAGIC 
# MAGIC The run time for the merging took about 1.5 hours, so the merged results were written to the storage for future read, so we wouldn’t need to rerun the merging. Although the writing process took about 7 hours, it turned out to be a wise choice as it made our feature engineering and modeling process later on more efficient. 

# COMMAND ----------

displayHTML('<iframe src="https://drive.google.com/file/d/18B2oMZRy5c7712H5tqIqoF0U65OXAGVn/preview" width="640" height="480" allow="autoplay"></iframe>')

# COMMAND ----------

# Create standardized gregorian timestamps for (FL_DATE:DEP_BLK_START, FL_DATE:DEP_BLK_END) and (FL_DATE:ARR_BLK_START, FL_DATE:ARR_BLK_END)
# FL_DATE is a standard DateType, TIME_BLK is a non-standard indicator of time start/end
# 2015-01-01T00:00:00.000+0000
from functools import partial
from datetime import datetime, timedelta

def get_timestamp(date, hour, idx, window_size_hours):
  _hours = hour.split('-')
  start = _hours[idx][:2]
  end = _hours[idx][2:]
  _ts = datetime.fromisoformat(f"{date}T{start}:{end}:00.000+00:00") - (timedelta(hours=1) if idx else timedelta(hours=window_size_hours))
  return _ts.isoformat()

def augment_flights_for_weather_join(df_flights, df_airports, window_size_hours=10):
  get_ts_low_udf = udf(partial(get_timestamp, idx=0, window_size_hours=window_size_hours))
  get_ts_high_udf = udf(partial(get_timestamp, idx=1, window_size_hours=window_size_hours))
  df_augmented = df_flights.withColumn("OriginWeatherStart", to_timestamp(get_ts_low_udf(df_flights.FL_DATE, df_flights.DEP_TIME_BLK)))\
                           .withColumn("OriginWeatherEnd", to_timestamp(get_ts_high_udf(df_flights.FL_DATE, df_flights.DEP_TIME_BLK)))\
                           .withColumn("DestWeatherStart", to_timestamp(get_ts_low_udf(df_flights.FL_DATE, df_flights.ARR_TIME_BLK)))\
                           .withColumn("DestWeatherEnd", to_timestamp(get_ts_high_udf(df_flights.FL_DATE, df_flights.ARR_TIME_BLK)))\
                           .withColumn("FlightID", monotonically_increasing_id())\
                           .join(df_airports.alias("origin")\
                                 .withColumnRenamed("StationID", "OriginStationID")\
                                 .withColumnRenamed("AirportLat", "OriginAirportLat")\
                                 .withColumnRenamed("AirportLon", "OriginAirportLon"),
                                 df_flights.ORIGIN == col("origin.AirportID"), 'inner')\
                           .join(df_airports.alias("dest")\
                                 .withColumnRenamed("StationID", "DestStationID")\
                                 .withColumnRenamed("AirportLat", "DestAirportLat")\
                                 .withColumnRenamed("AirportLon", "DestAirportLon"),
                                 df_flights.DEST == col("dest.AirportID"), 'inner')\
                           .drop("AirportID", "Haversine")
  return df_augmented

def build_training_dataframe(df_flights, df_airports_and_stations, df_weather):
  flight_data_cols = ['FlightID', 
                      'OriginStationID', 'OriginAirportLat', 'OriginAirportLon',
                      'DestStationID', 'DestAirportLat', 'DestAirportLon',
                      'YEAR','MONTH', 'DAY_OF_WEEK', 'FL_DATE','OP_UNIQUE_CARRIER','ORIGIN', 'DEST', 'IS_RushMonth',
                      'CRS_DEP_TIME','DEP_TIME','IS_DEP_RUSH_HOUR','DEP_DELAY','DEP_TIME_BLK','DEP_DEL15',
                      'CRS_ARR_TIME','ARR_TIME','IS_ARR_RUSH_HOUR','ARR_DELAY','ARR_TIME_BLK','ARR_DEL15',
                      'CRS_ELAPSED_TIME','DISTANCE',
                      'CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
  _airports_aug = augment_flights_for_weather_join(df_flights, df_airports_and_stations)
  df_flights_and_weather = _airports_aug.join(df_weather.alias("origin")\
                                              .withColumnRenamed("WND_ANGLE", "ORIGIN_WND_ANGLE")\
                                              .withColumnRenamed("WND_SPEED", "ORIGIN_WND_SPEED")\
                                              .withColumnRenamed("CIG_HEIGHT", "ORIGIN_CIG_HEIGHT")\
                                              .withColumnRenamed("VIS_DIST", "ORIGIN_VIS_DIST")\
                                              .withColumnRenamed("TMP_F", "ORIGIN_TMP_F")\
                                              .withColumnRenamed("DEW_0", "ORIGIN_DEW_0")\
                                              .withColumnRenamed("SLP_0", "ORIGIN_SLP_0"),
                                              on=[(_airports_aug.OriginStationID == col("origin.STATION")) &
                                                  (_airports_aug.OriginWeatherStart <= col("origin.DATE")) &
                                                  (_airports_aug.OriginWeatherEnd >= col("origin.DATE"))], 
                                                  how="leftouter")\
                                        .join(df_weather.alias("dest")\
                                              .withColumnRenamed("WND_ANGLE", "DEST_WND_ANGLE")\
                                              .withColumnRenamed("WND_SPEED", "DEST_WND_SPEED")\
                                              .withColumnRenamed("CIG_HEIGHT", "DEST_CIG_HEIGHT")\
                                              .withColumnRenamed("VIS_DIST", "DEST_VIS_DIST")\
                                              .withColumnRenamed("TMP_F", "DEST_TMP_F")\
                                              .withColumnRenamed("DEW_0", "DEST_DEW_0")\
                                              .withColumnRenamed("SLP_0", "DEST_SLP_0"),
                                              on=[(_airports_aug.DestStationID == col("dest.STATION")) &
                                                  (_airports_aug.DestWeatherStart <= col("dest.DATE")) &
                                                  (_airports_aug.DestWeatherEnd >= col("dest.DATE"))],
                                                  how="leftouter")\
                                        .groupby(flight_data_cols)\
                                        .mean("ORIGIN_WND_ANGLE", "ORIGIN_WND_SPEED", "ORIGIN_CIG_HEIGHT", "ORIGIN_VIS_DIST", "ORIGIN_TMP_F", "ORIGIN_DEW_0", "ORIGIN_SLP_0",
                                              "DEST_WND_ANGLE", "DEST_WND_SPEED", "DEST_CIG_HEIGHT", "DEST_VIS_DIST", "DEST_TMP_F", "DEST_DEW_0", "DEST_SLP_0")\
                                        .withColumnRenamed("avg(ORIGIN_WND_ANGLE)","DEP_WND_ANGLE")\
                                        .withColumnRenamed("avg(ORIGIN_WND_SPEED)","DEP_WND_SPEED")\
                                        .withColumnRenamed("avg(ORIGIN_CIG_HEIGHT)","DEP_CIG_HEIGHT")\
                                        .withColumnRenamed("avg(ORIGIN_VIS_DIST)","DEP_VIS_DIST")\
                                        .withColumnRenamed("avg(ORIGIN_TMP_F)","DEP_TMP_F")\
                                        .withColumnRenamed("avg(ORIGIN_DEW_0)","DEP_DEW_0")\
                                        .withColumnRenamed("avg(ORIGIN_SLP_0)","DEP_SLP_0")\
                                        .withColumnRenamed("avg(DEST_WND_ANGLE)","ARR_WND_ANGLE")\
                                        .withColumnRenamed("avg(DEST_WND_SPEED)","ARR_WND_SPEED")\
                                        .withColumnRenamed("avg(DEST_CIG_HEIGHT)","ARR_CIG_HEIGHT")\
                                        .withColumnRenamed("avg(DEST_VIS_DIST)","ARR_VIS_DIST")\
                                        .withColumnRenamed("avg(DEST_TMP_F)","ARR_TMP_F")\
                                        .withColumnRenamed("avg(DEST_DEW_0)","ARR_DEW_0")\
                                        .withColumnRenamed("avg(DEST_SLP_0)","ARR_SLP_0")
  return df_flights_and_weather


df_merged = build_training_dataframe(airlines_preprocessed,df_nearest,weather_preprocessed)
display(df_merged.head(5))

# COMMAND ----------

displayHTML('<iframe src="https://drive.google.com/file/d/1LgJIxzIk3Ya_V5ZohDgbT1_TtwuoTbSz/preview" width="640" height="480" allow="autoplay"></iframe>')

# COMMAND ----------

# MAGIC %md The image above is a high-level illustration of the path we took to join the various datasets together.

# COMMAND ----------

# Save merged dataset to the local storage as parquets
df_merged.write.parquet(f"{blob_url}/df_merged_full_0731_2.parquet")

# COMMAND ----------

# Read locally saved dataset
df_merged = spark.read.parquet(f"{blob_url}/df_merged_full_0731_2.parquet")

# COMMAND ----------

# MAGIC %md #####3.5 Check Missing Values
# MAGIC The merged dataset did not include any NULLs, so no missing value handling was required. 

# COMMAND ----------

# create a function to show missing values:
def nullDataFrame(df):
  '''
  Returns a pandas dataframe consisting of column names, null values and percentage of null values for the given datftame 
  '''
  null_feature_list = []
  count = df.count()
  for column in df.columns:
    nulls = df.filter(df[column].isNull()).count()
    nulls_perct = np.round((nulls/count)*100, 2)
    null_feature_list.append([column, nulls, nulls_perct])
  nullCounts_df = pd.DataFrame(np.array(null_feature_list), columns=['Feature_Name', 'Null_Counts', 'Percentage_Null_Counts'])
  return nullCounts_df

nullCounts_df = nullDataFrame(df_merged)
display(nullCounts_df)

# COMMAND ----------

# MAGIC %md #####3.6 Encoding, Grouping And Dummy Variables
# MAGIC 
# MAGIC At this stage, we began to start encoding information as numerical representations for categories. We started by coding time blocks of the day to numerical representations using a dictionary.
# MAGIC 
# MAGIC Next we deal with a categorical column that contains a large amount of unique values, Airports. Airports contains around 317 unique values, so to represent these points in our model, we will group airports based off geographical location. The groups are split into 3 sections: West, Mid, East. The section in which an airport belongs is determined by the airport's longitudinal coordinate. All airports greater than 110 fall into the West section. Airports under 90 fall under the east category. All else falls in the Mid section.
# MAGIC 
# MAGIC Lastly, since we wanted to see the effects of each individual carrier, we used dummy variable representation for all of the carrier ids.

# COMMAND ----------

# Encode DEP_TIME_BLOCK and ARR_TIME_BLOCK

dict = {'0001-0559':0, '0600-0659':1, '0700-0759':2, '0800-0859':3, '0900-0959':4, '1000-1059':5, '1100-1159':6, '1200-1259':7, '1300-1359':8, '1400-1459':9, '1500-1559':10, '1600-1659':11, '1700-1759':12, '1800-1859':13, '1900-1959':14, '2000-2059':15, '2100-2159':16, '2200-2259':17, '2300-2359':18}
user_func =  udf (lambda x: dict.get(x), IntegerType())

newdf = df_merged.withColumn('DEP_TIME_BLK',user_func(df_merged.DEP_TIME_BLK))
newdf = newdf.withColumn('ARR_TIME_BLK',user_func(newdf.ARR_TIME_BLK))
display(newdf.head(5))

# COMMAND ----------

# Group ORINGIN/DEST into WEST/MID/EAST by longitude

def location_split(longitude, section):
    "Returns 0 or 1 if longitude matches stated section limits"
    if section.lower() == "west":
        if longitude < float(-110):   # west coast break 110 degree
            return 1
        else:
            return 0
    elif section.lower() == "mid":
        if longitude > float(-110) and longitude < float(-90): # mid us break 110 degree - 90 degree
            return 1
        else:
            return 0
    elif section.lower() == "east":
        if longitude > float(-90): # east coast break 90 degree
            return 1
        else:
            return 0

west = udf(partial(location_split, section="west"))
mid = udf(partial(location_split, section="mid"))
east = udf(partial(location_split, section="east"))

mod_df = newdf.withColumn("WestOrigin", west(newdf.OriginAirportLon))\
              .withColumn("MidOrigin", mid(newdf.OriginAirportLon))\
              .withColumn("EastOrigin", east(newdf.OriginAirportLon))\
              .withColumn("WestDest", west(newdf.DestAirportLon))\
              .withColumn("MidDest", mid(newdf.DestAirportLon))\
              .withColumn("EastDest", east(newdf.DestAirportLon))

mod_df = mod_df.withColumn("EastDest", mod_df["EastDest"].cast(IntegerType()))\
               .withColumn("EastOrigin", mod_df["EastOrigin"].cast(IntegerType()))\
               .withColumn("WestDest", mod_df["WestDest"].cast(IntegerType()))\
               .withColumn("WestOrigin", mod_df["WestOrigin"].cast(IntegerType()))\
               .withColumn("MidDest", mod_df["MidDest"].cast(IntegerType()))\
               .withColumn("MidOrigin", mod_df["MidOrigin"].cast(IntegerType()))\

display(mod_df.head(5))

# COMMAND ----------

# Making dummy variables for airline carriers

def create_dummies(df, dummylist):
  
  for inputcol in dummylist:
    categories = df.select(inputcol).rdd.distinct().flatMap(lambda x: x).collect()

    exprs = [f.when(f.col(inputcol) == category, 1).otherwise(0).alias(category) for category in categories]
    for index,column in enumerate(exprs):
      df=df.withColumn(categories[index], column)

  return df

mod_df2 = create_dummies(mod_df, ["OP_UNIQUE_CARRIER"])
display(mod_df2.limit(5))

# COMMAND ----------

# MAGIC %md #####3.7 Split The Dataset & Undersampling 
# MAGIC 
# MAGIC Now that the dataset contains the information that is desired for building the dataset, we split the data into training and testing. This split is based on year; we selected all of 2019 as the test year and the 2015~2018 data as the training set.
# MAGIC 
# MAGIC In order to deal with the previously seen imbalance with the label, ARR_DELAY, we need to pre-process the training dataset so that the delay records are not diluted by the non-delay records. To achieve this solution, we decided to undersample the non-delay instances in the training set to match that of the delay instances. This selection of the non-delay instances is done randomly.

# COMMAND ----------

# Split the set: 2015~2018 as Train, 2019 as Test
train = mod_df2.filter(mod_df2.YEAR != 2019)
test = mod_df2.filter(mod_df2.YEAR == 2019)

# COMMAND ----------

# Resample Train set to make delayed/undelayed as 50%/50%

def resample(df, ratio, class_field, base_class):
    late = df.filter(col(class_field)==base_class)
    not_late = df.filter(col(class_field)!=base_class)
    total_pos = late.count()
    total_neg = not_late.count()
    fraction=float(total_pos*ratio)/float(total_neg)
    sampled = not_late.sample(False, fraction)
    
    return sampled.union(late)

train_resampled = resample(train, 1, 'ARR_DEL15', 1)
display(train_resampled.head(5))

# COMMAND ----------

# MAGIC %md #####3.8 Normalization
# MAGIC 
# MAGIC For numerical columns in the train set, a MinMaxScaler was used to normalize them so that they are all on the same scale and therefore have the same effect on the model. Then the scaler from the train set normalization was used to normalize the test set. 
# MAGIC 
# MAGIC At this point, we got a normalized train set and test set that were ready to be used for the modeling process.

# COMMAND ----------

# Normalize Train set, then use the scaler to normalize Test set

numerical_col=['DEP_DELAY','ARR_DELAY','CRS_ELAPSED_TIME','DISTANCE','DEP_WND_ANGLE','DEP_WND_SPEED','DEP_CIG_HEIGHT','DEP_VIS_DIST','DEP_TMP_F','DEP_DEW_0','DEP_SLP_0','ARR_WND_ANGLE','ARR_WND_SPEED','ARR_CIG_HEIGHT','ARR_VIS_DIST','ARR_TMP_F','ARR_DEW_0','ARR_SLP_0']

def TrainNormalizer(df_train, col_to_scale):
  '''
  Normalize columns in col_to_scale, and append the normalized columns into train df
  '''
  assembler = VectorAssembler(inputCols=col_to_scale, outputCol="_Vect")
  temp_train = assembler.transform(df_train)
  # MinMaxScaler Transformation
  scaler = MinMaxScaler(inputCol="_Vect", outputCol="_SCALED")
  # Fitting training set
  train_fit = scaler.fit(temp_train)
  # Transforming test set to training set scale
  df = train_fit.transform(temp_train)
  for i in range(len(col_to_scale)):
    unlist = udf(lambda x: float(list(x)[i]), DoubleType())
    df = df.withColumn(col_to_scale[i]+"_SCALED", unlist("_SCALED"))
  df = df.drop("_Vect").drop("_SCALED")
  return df

train_normalized = TrainNormalizer(train_resampled,numerical_col).cache()
#display(train_normalized.head(5))

def TestNormalizer(df_train, df_test, col_to_scale):
  '''
  Normalize columns in col_to_scale, and append the normalized columns into test df
  based off the values in the train df
  '''
  assembler = VectorAssembler(inputCols=col_to_scale, outputCol="_Vect")
  temp_train = assembler.transform(df_train)
  temp_test = assembler.transform(df_test)
  # MinMaxScaler Transformation
  scaler = MinMaxScaler(inputCol="_Vect", outputCol="_SCALED")
  # Fitting training set
  train_fit = scaler.fit(temp_train)
  # Transforming test set to training set scale
  df = train_fit.transform(temp_test)
  for i in range(len(col_to_scale)):
    unlist = udf(lambda x: float(list(x)[i]), DoubleType())
    df = df.withColumn(col_to_scale[i]+"_SCALED", unlist("_SCALED"))
  df = df.drop("_Vect").drop("_SCALED")
  return df

test_normalized = TestNormalizer(train_resampled, test, numerical_col).cache()
#display(test_nomarlized.head(5))

# COMMAND ----------

# MAGIC %md ## 4. Algorithm Theory: XGBoost 

# COMMAND ----------

# MAGIC %md We explained the math of XGBoost. Please see this [Google Doc](https://docs.google.com/document/d/1u4RDmSqHKss3-KTYzelkwpoYJeixz_CoQhzZuOy04rM/edit?usp=sharing) for details.

# COMMAND ----------

# MAGIC %md ## 5. Implementation

# COMMAND ----------

# MAGIC %md #####5.0 Feature & Matrix Selection
# MAGIC 
# MAGIC ARR_DEL15 is selected as the outcome variable. Below features will be used to predict the outcome variable:
# MAGIC * flight descriptive features: MONTH, DAY_OF_WEEK, DISTANCE
# MAGIC * rush month and rush intraday hour blocks: 'IS_DEP_RUSH_HOUR','IS_ARR_RUSH_HOUR','IS_RushMonth'
# MAGIC * departure/arrival airport location: WEST/MID/EAST
# MAGIC * departure/arrival airport weather data
# MAGIC * carrier
# MAGIC 
# MAGIC Precision and Recall will be the main matrix to evaluate the model performance. From passenger perspective, they would care more about Precision; from airline company perspective, they would care more about Recall. But we will also display F1 and Accuracy of each model as reference.
# MAGIC 
# MAGIC Based off of the nature of the problem and our knowledge of the following algorithms, we expect Logistic Regression to perform the worse in terms of our matrix. The problem on hand is not a linear problem and regression works best with linear issues. Therefore we expect this implementation would not work as well for predicting arrival delays. Decision tree and random forest, we expect to have similar performance. But due to the nature of the random forest algorithm, this implementation should have a better score since this algorithm usually performs better with generalization since it doesn't completely focus on specific features for importance. Finally the best performing model is expected to be XGBoost, but due to the nature of this algorithm, it is expected to take the longest to train. 

# COMMAND ----------

# Select columns to be used in the modeling

label = "ARR_DEL15"
featuresCols = ['ARR_DEL15',
            'MONTH','DAY_OF_WEEK','DISTANCE_SCALED',
            'IS_DEP_RUSH_HOUR','IS_ARR_RUSH_HOUR','IS_RushMonth',
            'DEP_TIME_BLK','ARR_TIME_BLK',                
            'WestOrigin','WestDest',
            'MidOrigin','MidDest',
            'EastOrigin','EastDest',
            'DEP_WND_ANGLE_SCALED','DEP_WND_SPEED_SCALED','DEP_CIG_HEIGHT_SCALED','DEP_TMP_F_SCALED','DEP_VIS_DIST_SCALED','DEP_DEW_0_SCALED','DEP_SLP_0_SCALED',
            'ARR_WND_ANGLE_SCALED','ARR_WND_SPEED_SCALED', 'ARR_CIG_HEIGHT_SCALED','ARR_TMP_F_SCALED','ARR_VIS_DIST_SCALED','ARR_DEW_0_SCALED','ARR_SLP_0_SCALED',
            'UA','F9','MQ','WN','DL','OO','AS','G4','B6','9E','YX','NK','YV','US','AA','EV','VX','OH']


featuresCols.remove(label)
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")

# COMMAND ----------

# Create a function to give evaluation matrix

def EvaluationMatrix(prediction, eval_label):
  
  matrix_name = ['Precision','Recall','Accuracy','F1']
  results = []
  
  matrix = ['precisionByLabel','recallByLabel','accuracy','f1']
  for i in range(len(matrix)):
    evaluator = MulticlassClassificationEvaluator()
    evaluator.setMetricName(matrix[i])
    evaluator.setBeta(1)
    evaluator.setLabelCol(eval_label)
    evaluator.setPredictionCol("prediction")
    result = evaluator.evaluate(prediction)
    results.append(result)
  
  df = pd.DataFrame([matrix_name, results], index=['Matrix', 'Value']).T.explode('Value')
  return df

# COMMAND ----------

# MAGIC %md #####5.1 Model 1: Logistic Regression -- Baseline Model
# MAGIC 
# MAGIC We used a plain-vanilla LogisticRegression as our baseline model at this stage without any optimization. It gives a 87.57% precision and 60.07% recall, with 60.54% accuracy and 64.91% f1. The training time was less than 10 minutes. 

# COMMAND ----------

# LR Model 1: plain-vanilla LogisticRegression

lr_1 = LogisticRegression(labelCol = label, maxIter=10)

pipeline_lr_1 = Pipeline(stages=[vectorAssembler, lr_1])
lrModel_1 = pipeline_lr_1.fit(train_normalized)
predictions_lr_1 = lrModel_1.transform(test_normalized)

# COMMAND ----------

# Evaluate LR Model 1
matrix_lr_1 = EvaluationMatrix(predictions_lr_1, 'ARR_DEL15')
display(matrix_lr_1)

# COMMAND ----------

# MAGIC %md #####5.2 Model 2: Logistic Regression -- Enet + GridSearch + CrossValidation
# MAGIC 
# MAGIC In this second LogisticRegression, elasticNetParam was set to 0.5 to add both L1 and L2 regularization. 4-folds cross validation was also used with the hope to improve the performance. It gives a 88.06% precision and 56.78% recall, with 58.54% accuracy and 63.14% f1.

# COMMAND ----------

# LR Model 2: ENET + GridSearch + CrossValidation

lr_2 = LogisticRegression(labelCol = label, maxIter=10, elasticNetParam=0.5)

paramGrid_lr = ParamGridBuilder() \
              .addGrid(lr_2.regParam, [0.1, 0.01]) \
              .build()

cv_lr = CrossValidator(estimator=lr_2, evaluator=MulticlassClassificationEvaluator().setLabelCol("ARR_DEL15"), estimatorParamMaps=paramGrid_lr,numFolds=4)

pipeline_lr_2 = Pipeline(stages=[vectorAssembler, cv_lr])
lrModel_2 = pipeline_lr_2.fit(train_normalized)
predictions_lr_2 = lrModel_2.transform(test_normalized)

# COMMAND ----------

matrix_lr_2 = EvaluationMatrix(predictions_lr_2, 'ARR_DEL15')
display(matrix_lr_2)

# COMMAND ----------

# MAGIC %md #####5.3 Model 3: Decision Tree -- GridSearch + CrossValidation
# MAGIC 
# MAGIC DecisionTree is our first tree-based model. maxDepth determines the maximum depth of each tree; when this is too small, the trained random forest may have a high bias; if this is too large, the trained random forest may have a high variance. 4-folds cross validation was also used in this model. 
# MAGIC 
# MAGIC It gives a 88.25% precision and 59.18% recall, with 60.35% accuracy and 64.75% f1. The training time was around 10 to 15 minutes. 

# COMMAND ----------

# Model 3: Decision Tree

dt = DecisionTreeClassifier(labelCol=label, featuresCol='features')

paramGrid_dt = ParamGridBuilder()\
    .addGrid(dt.maxDepth, [2,6])\
    .build()

cv_dt = CrossValidator(estimator=dt, 
                      evaluator=MulticlassClassificationEvaluator().setLabelCol("ARR_DEL15"),
                      estimatorParamMaps=paramGrid_dt,
                      numFolds=4)

pipeline_dt = Pipeline(stages=[vectorAssembler, cv_dt])
dtModel = pipeline_dt.fit(train_normalized)
predictions_dt = dtModel.transform(test_normalized)

# COMMAND ----------

matrix_dt = EvaluationMatrix(predictions_dt, 'ARR_DEL15')
display(matrix_dt)

# COMMAND ----------

# MAGIC %md #####5.4 Model 4: Random Forest -- GridSearch + CrossValidation
# MAGIC 
# MAGIC RamdonForest is our second tree-based model. numTrees was used to determine how many decision trees are trained using the bootstrapping process; more trees, lower variance. maxDepth determines the maximum depth of each tree; when this is too small, the trained random forest may have a high bias; if this is too large, the trained random forest may have a high variance. 4-folds cross validation was also used in this model. 
# MAGIC 
# MAGIC It gives a 89.42% precision and 64.69%, with 65.01% accuracy and 68.81% f1. It took about 50 minutes to train.  

# COMMAND ----------

# Model 4: Random Forest

rf = RandomForestClassifier(labelCol=label, featuresCol="features")

paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [7]) \
    .build()

cv_rf = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid_rf,
    evaluator=MulticlassClassificationEvaluator().setLabelCol("ARR_DEL15"),
    numFolds=4) # 4 Folds

pipeline_rf = Pipeline(stages=[vectorAssembler, cv_rf])
rfModel = pipeline_rf.fit(train_normalized)
predictions_rf = rfModel.transform(test_normalized)

# COMMAND ----------

matrix_rf = EvaluationMatrix(predictions_rf, 'ARR_DEL15')
display(matrix_rf)

# COMMAND ----------

# MAGIC %md #####5.5 Model 5: XGBoost -- GridSearch + CrossValidation
# MAGIC 
# MAGIC XGBoost is our third tree-based model. n_estimators sets the number of trees or rounds in a XGB. maxDepth determines the maximum depth of each tree; XGB generally uses simple shallow trees with one or two splits but uses thousands of trees to form a boosted tree. Again, 4-folds cross validation was also used in this model. 
# MAGIC 
# MAGIC It gives a 89.89% precision and 59.18% recall, with 73.42% accuracy and 75.68% f1. The training time was around 4 hours, potentially because of the high number of n_estimators.

# COMMAND ----------

# Model 5: XGBoost

xgb = XgboostClassifier(labelCol=label, missing=0.0)

paramGrid_xgb = ParamGridBuilder()\
  .addGrid(xgb.max_depth, [4, 6])\
  .addGrid(xgb.n_estimators, [50, 100])\
  .build()

cv_xgb = CrossValidator(estimator=xgb, evaluator=MulticlassClassificationEvaluator().setLabelCol("ARR_DEL15"), estimatorParamMaps=paramGrid_xgb, numFolds=4)
pipeline_xgb = Pipeline(stages=[vectorAssembler, cv_xgb])
xgbModel = pipeline_xgb.fit(train_normalized)
predictions_xgb = xgbModel.transform(test_normalized)

# COMMAND ----------

matrix_xgb = EvaluationMatrix(predictions_xgb, 'ARR_DEL15')
display(matrix_xgb)

# COMMAND ----------

# MAGIC %md ## 6. Conclusion

# COMMAND ----------

models = ['LogisticRegression_1','LogisticRegression_2','DecisionTree','RandomForest','XGBoost']
precisions = ['87.57%','88.06%','88.25%','89.42%','89.89%']
recalls = ['60.07%','56.78%','59.18%','64.69%','75.89%']
f1s = ['64.91%','63.14%','64.75%','68.81%','75.68%']
accuracies = ['60.54%','58.54%','60.35%','65.01%','73.42%']
df_matrix = pd.DataFrame({'Model': models, 'Precision': precisions, 'Recall': recalls, 'F1': f1s, 'Accuracy': accuracies} )
HTML(df_matrix.to_html(index=False))

# COMMAND ----------

# MAGIC %md 
# MAGIC * As hypothsized, XGBoost performed the best, not only in our chosen metrics but in every metric we measured. 
# MAGIC * Although XGBoost performed the best, it took the longest to train; more than 4 times the length of the second longest algorithm, random forest. 
# MAGIC * The final implementation of XGBoost gave us a percision score of 89.89% and a recall score of 75.89%. The random forest implementation came at a close second in percision, 89.42%, but not as close in recall, 64.69%. Although random forest didn't perform as well in recall as XGBoost, it is a better algorithm for implementation in a cluster; that can be seen in the training time.

# COMMAND ----------

# MAGIC %md ## 7. Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC **Normalization**: We applied min/max scalers functions to normalize multi-range numerical data in both the flight and weather data to contrarest range influence on training, and therefore prediction. 
# MAGIC 
# MAGIC **Vector Embeddings**: We took advantage of the VectorAssembler functionality in order to combine our features list into a single vector column. This representation helps the model find similarities as per proximity in the vector space.
# MAGIC 
# MAGIC **I/O vs Memory**:  I/O scaled well in the Spark Databricks cluster. It only took ~1.5hrs to join Airlines/Weather on our massive combined  dataset. The classifiers also scaled well with the merged datasets; Logistic Regression and Decision Tree trained under 10 minutes. Pre-writing the merged dataset also improved the efficiency of our feature engineering and modeling process.
# MAGIC 
# MAGIC **Assumptions**: Assumptions of different algorithms played an important role during our EDA and feature engineering. For example, we have to make sure features used in Logistic Regression have minimal multicollinearity , so we need to be careful about feature selections.  Whereas Decision Trees are much less sensitive to highly correlated features and therefore make us less concerned about feature selection.
# MAGIC 
# MAGIC **Regularization**: We used Enet in our second Logistic Regression model. It combines both L1 and L2 regularization. Enet, together with GridSearch and CrossValidation allowed us to better tune the models and improve our precision and recall

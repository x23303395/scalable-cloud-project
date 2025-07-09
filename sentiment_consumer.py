from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, window, length, trim
from pyspark.sql.types import StructType, StringType, FloatType
from textblob import TextBlob

# Spark session with tuned configs
spark = SparkSession.builder \
    .appName("StreamingSentimentAnalysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Kafka schema
schema = StructType().add("book", StringType()).add("review", StringType())

# Kafka stream read
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "10.11.11.125:9092") \
    .option("subscribe", "book_reviews") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse and clean JSON
json_df = df.selectExpr("CAST(value AS STRING)", "timestamp")
parsed_df = json_df.select(from_json(col("value"), schema).alias("data"), "timestamp") \
    .select("data.book", "data.review", "timestamp") \
    .filter(col("review").isNotNull()) \
    .filter(length(col("review")) > 3)

# Sentiment UDF using TextBlob
def get_sentiment(text):
    try:
        return float(TextBlob(text).sentiment.polarity)
    except:
        return 0.0

sentiment_udf = udf(get_sentiment, FloatType())

# Apply sentiment analysis
sentiment_df = parsed_df.withColumn("sentiment", sentiment_udf(trim(col("review"))))

# Time-windowed sentiment aggregation
aggregated_sentiment = sentiment_df \
    .withWatermark("timestamp", "5 minutes") \
    .groupBy(window(col("timestamp"), "3 minutes", "1 minute"), col("book")) \
    .agg({"sentiment": "avg"}) \
    .withColumnRenamed("avg(sentiment)", "avg_sentiment")

# Write stream output
def write_to_parquet_sentiment(batch_df, epoch_id):
    batch_df.persist()
    batch_df.write.mode("append").parquet("/home/ubuntu/sentiment_output_parquet/")
    batch_df.unpersist()

query = aggregated_sentiment.writeStream \
    .outputMode("update") \
    .foreachBatch(write_to_parquet_sentiment) \
    .option("checkpointLocation", "/home/ubuntu/sentiment_checkpoints/") \
    .trigger(processingTime="1 minute") \
    .start()

query.awaitTermination()

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, explode, split, window, length, lower, trim
from pyspark.sql.types import StructType, StringType

# Initialize Spark session with memory and performance tuning
spark = SparkSession.builder \
    .appName("StreamingReviewsWithWatermark") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.streaming.backpressure.enabled", "true") \
    .config("spark.sql.streaming.stateStore.maintenanceInterval", "300s") \
    .config("spark.sql.streaming.metricsEnabled", "true") \
    .getOrCreate()

# Schema for incoming data
schema = StructType().add("book", StringType()).add("review", StringType())

# Read stream from Kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "10.11.11.125:9092") \
    .option("subscribe", "book_reviews") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse Kafka JSON and extract fields
json_df = df.selectExpr("CAST(value AS STRING)", "timestamp")
parsed_df = json_df.select(from_json(col("value"), schema).alias("data"), "timestamp") \
    .select("data.book", "data.review", "timestamp")

# Split and clean words
words = parsed_df.select(
    explode(split(col("review"), r"\s+")).alias("word"),
    col("timestamp")
).withColumn("word", lower(trim(col("word")))) \
 .filter(length("word") > 1) \
 .filter(~col("word").isin(
    "the", "and", "a", "an", "is", "in", "it", "to", "of", "for", "on", "that", "this", "with", "as", "was", "were"
))

# Reduce window + watermark retention to lower memory use
word_counts = words \
    .withWatermark("timestamp", "3 minutes") \
    .groupBy(window(col("timestamp"), "2 minutes", "1 minute"), col("word")) \
    .count()

# Write each micro-batch to Parquet
def write_to_parquet(batch_df, epoch_id):
    batch_df.persist()
    batch_df.write.mode("append").parquet("/home/ubuntu/stream_output_parquet/")
    batch_df.unpersist()

query = word_counts.writeStream \
    .outputMode("update") \
    .foreachBatch(write_to_parquet) \
    .option("checkpointLocation", "/home/ubuntu/stream_checkpoints_optimized/") \
    .trigger(processingTime="1 minute") \
    .start()

query.awaitTermination()

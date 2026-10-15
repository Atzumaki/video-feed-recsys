"""Example Spark job: sessionize raw events and build point-in-time features."""


def main() -> None:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as functions

    spark = SparkSession.builder.appName("video-feed-features").getOrCreate()
    events = spark.read.parquet("hdfs:///events/video_feed/date=*/")
    window = (
        Window.partitionBy("user_id")
        .orderBy(functions.col("event_ts").cast("long"))
        .rangeBetween(-7 * 24 * 3600, -1)
    )
    features = (
        events.withColumn("watches_7d", functions.sum("is_watch").over(window))
        .withColumn("purchases_7d", functions.sum("is_purchase").over(window))
        .withColumn("mean_watch_ms_7d", functions.avg("watch_ms").over(window))
        .repartition("event_date")
    )
    features.write.mode("overwrite").partitionBy("event_date").parquet(
        "hdfs:///features/video_feed/"
    )
    spark.stop()


if __name__ == "__main__":
    main()


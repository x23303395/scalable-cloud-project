import logging
import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def produce_reviews_in_chunks(csv_path, kafka_servers, topic_name,
                               chunk_size=200, sleep_sec=0.2, max_messages=50000):
    """
    Produce messages to Kafka in controlled batches with an upper message limit.
    """
    try:
        reader = pd.read_csv(csv_path, engine='python', chunksize=chunk_size, iterator=True)
        logging.info(f"Initialized CSV reader with chunk size {chunk_size}")
    except Exception as e:
        logging.error(f"Failed to open CSV file: {e}")
        return

    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=5
        )
        logging.info(f"Connected to Kafka servers: {kafka_servers}")
    except Exception as e:
        logging.error(f"Failed to create Kafka producer: {e}")
        return

    total_sent = 0
    chunk_number = 0

    for chunk in reader:
        if total_sent >= max_messages:
            break

        chunk_number += 1
        chunk.columns = chunk.columns.str.strip()
        sent_in_chunk = 0

        for idx, row in chunk.iterrows():
            if total_sent >= max_messages:
                break

            try:
                review = {
                    'book': str(row['Title']),
                    'review': str(row['review/text'])
                }
                producer.send(topic_name, value=review)
                sent_in_chunk += 1
                total_sent += 1

                if total_sent % 1000 == 0:
                    logging.info(f"Progress: Sent {total_sent} messages")

            except KafkaError as ke:
                logging.error(f"Kafka error in chunk {chunk_number}, row {idx}: {ke}")
            except Exception as e:
                logging.error(f"Error in chunk {chunk_number}, row {idx}: {e}")

        try:
            producer.flush()
            logging.info(f"Chunk {chunk_number}: Sent {sent_in_chunk} messages (total: {total_sent})")
        except Exception as e:
            logging.error(f"Error flushing producer at chunk {chunk_number}: {e}")

        time.sleep(sleep_sec)

    logging.info(f"Finished sending {total_sent} messages.")
    producer.close()
    logging.info("Kafka producer closed")


if __name__ == "__main__":
    CSV_FILE = "merged_books_data.csv"
    KAFKA_SERVERS = ['10.11.11.125:9092']
    TOPIC = "book_reviews"

    # You can adjust these limits per your memory constraints
    produce_reviews_in_chunks(
        csv_path=CSV_FILE,
        kafka_servers=KAFKA_SERVERS,
        topic_name=TOPIC,
        chunk_size=200,
        sleep_sec=0.2,
        max_messages=50000  # cap total messages sent
    )

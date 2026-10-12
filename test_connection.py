import os
from dotenv import load_dotenv
from databricks import sql

load_dotenv()  # read the values from .env

with sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_TOKEN"),
) as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM spacewatch.gold.neo_embeddings")
        print("Rows in embeddings table:", cursor.fetchone()[0])
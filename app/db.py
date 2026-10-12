import os
import pandas as pd
from dotenv import load_dotenv
from databricks import sql

load_dotenv()


def run_query(query):
    """Run a SQL query on Databricks and return the results as a pandas table."""
    with sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]   # column names
            return pd.DataFrame(cursor.fetchall(), columns=columns)
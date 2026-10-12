# SpaceWatch

Ask questions about near-Earth asteroids and get answers grounded in NASA data.

**[Live demo](https://spacewatch-neo.streamlit.app/)**

SpaceWatch pulls NASA's Near-Earth Object data every morning, processes it through a bronze → silver → gold lakehouse on Databricks, and serves it through a FastAPI service and a Streamlit web app.

## Features

- **Daily data refresh** of every asteroid close approach since January 2020 (~14,700 and counting)
- **Summary dashboard** for any date range: closest, fastest, and hazardous passes
- **Asteroid lookup**: click any asteroid to see its size, hazard status, and full pass history
- **Ask SpaceWatch**: plain-English questions answered by Claude, citing the exact NASA records used

## How it works

1. **Bronze:** raw NASA NeoWs responses land in a Databricks volume and load into Delta with Auto Loader
2. **Silver:** JSON is parsed, typed, and deduplicated into one row per close approach (idempotent `MERGE`)
3. **Gold:** summary tables, plus plain-English documents for each approach
4. **Embeddings:** each document is embedded with sentence-transformers for search
5. **Ask:** hybrid search (exact filters for dates, names, and "closer than the Moon," then meaning search) feeds Claude, which answers only from the retrieved records

The pipeline runs daily as a Databricks Job, with an Airflow DAG that can also trigger it.

## Tech stack

Databricks (Unity Catalog, Delta Lake, Auto Loader) · PySpark · Apache Airflow · sentence-transformers · Claude · FastAPI · Streamlit

## Data notes

Data comes from [NASA NeoWs](https://api.nasa.gov). NeoWs isn't a complete record of every close approach: early 2020 shows unusually high volume, and very close passes are underrepresented. SpaceWatch reports what NeoWs returns.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your Databricks and Anthropic settings
uvicorn app.main:app --reload
```

---

Built by Christina Huynh

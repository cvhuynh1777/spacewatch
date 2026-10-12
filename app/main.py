import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.search import SearchEngine
from app.answer import write_answer
from app.db import run_query

# Holds the search engine after it loads (so it's built once, not on every request)
state = {}


@asynccontextmanager
async def lifespan(app):
    # Runs once when the server starts: load documents, embeddings, and the model
    state["engine"] = SearchEngine()
    yield
    # (anything after yield runs when the server shuts down; nothing needed here)


app = FastAPI(
    title="SpaceWatch API",
    description="Ask questions about near-Earth asteroid close approaches (NASA NeoWs data, 2020 to today).",
    lifespan=lifespan,
)


# ---------- Request and response shapes ----------
# Pydantic models define what JSON the API accepts and returns,
# and FastAPI validates it automatically.

class Filters(BaseModel):
    closer_than_moon: Optional[bool] = None   # only passes closer than the Moon
    is_hazardous: Optional[bool] = None       # only hazardous (true) or non-hazardous (false)
    year: Optional[int] = None                # only passes in this year


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Tell me about asteroid 2025 US6"])
    filters: Optional[Filters] = None


class Source(BaseModel):
    doc_id: str
    document: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    filters: dict
    sources: list[Source]


def table_to_json(df):
    """Convert a pandas table to plain JSON-friendly records (handles dates and numbers)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ---------- Endpoints ----------

@app.get("/health")
def health():
    """Check that the API is running and the documents are loaded."""
    return {"status": "ok", "documents_loaded": len(state["engine"].docs)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question using hybrid search plus Claude, with cited sources."""
    filters = request.filters.model_dump(exclude_none=True) if request.filters else {}

    documents = state["engine"].search(request.question, filters or None)
    answer = write_answer(request.question, documents)

    return {
        "question": request.question,
        "answer": answer,
        "filters": filters,
        "sources": [{**d, "score": float(d["score"])} for d in documents],
    }


@app.get("/stats/monthly")
def monthly_stats(year: Optional[int] = None):
    """Monthly counts and records from the gold summary table. Optionally filter by year."""
    query = "SELECT * FROM spacewatch.gold.neo_monthly_summary"
    if year:
        query += f" WHERE year(month) = {int(year)}"   # int() guarantees it's a number (no SQL injection)
    query += " ORDER BY month"
    return table_to_json(run_query(query))


@app.get("/stats/closest-by-year")
def closest_by_year():
    """The closest asteroid approach in each year."""
    return table_to_json(run_query("SELECT * FROM spacewatch.gold.neo_closest_by_year ORDER BY year"))
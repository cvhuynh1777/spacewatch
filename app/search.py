import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.db import run_query

# Must be the same model used to build the embeddings in Databricks
MODEL_NAME = "all-MiniLM-L6-v2"

# Asteroid names look like "2025 US6": 4 digits, a space, 2 letters, optional digits
ASTEROID_NAME = re.compile(r"\b(\d{4} [A-Z]{2}\d*)\b")


class SearchEngine:
    """Loads all documents + embeddings once, then answers searches from memory."""

    def __init__(self):
        # 1. Load documents and embeddings from the gold table
        print("Loading documents from Databricks...")
        self.docs = run_query("""
            SELECT doc_id, document, approach_date, is_hazardous, closer_than_moon, embedding
            FROM spacewatch.gold.neo_embeddings
        """)

        # Embeddings can arrive as text like "[0.1, 0.2, ...]"; turn them into real lists
        self.docs["embedding"] = self.docs["embedding"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else list(v)
        )
        self.vectors = np.array(self.docs["embedding"].tolist(), dtype=np.float32)

        # 2. Load the embedding model (used to encode questions)
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"Ready: {len(self.docs)} documents loaded.")

    def search(self, question, filters=None, top_k=8):
        """Find the documents closest in meaning to the question.
        filters: optional exact rules, e.g. {"closer_than_moon": True, "year": 2024}"""
        q = self.model.encode(question, normalize_embeddings=True)
        scores = self.vectors @ q                       # similarity to every document

        # Exact filters: give non-matching documents the lowest possible score
        if filters:
            for column, value in filters.items():
                if column == "year":
                    keep = pd.to_datetime(self.docs["approach_date"]).dt.year == int(value)
                else:
                    keep = self.docs[column] == value
                scores[~keep.to_numpy()] = -np.inf

        # Named asteroid: keep only its documents, and return all of its passes
        match = ASTEROID_NAME.search(question.upper())
        if match:
            name_in_doc = f"({match.group(1)})"
            has_name = self.docs["document"].str.contains(name_in_doc, regex=False).to_numpy()
            scores[~has_name] = -np.inf
            top_k = 30

        # Pick the best matches
        best = np.argsort(-scores)[:top_k]
        best = [i for i in best if np.isfinite(scores[i])]

        results = self.docs.iloc[best][["doc_id", "document"]].copy()
        results["score"] = scores[best]
        return results.to_dict(orient="records")       # list of {doc_id, document, score}
from app.search import SearchEngine
from app.answer import write_answer

engine = SearchEngine()

question = "Tell me about asteroid 2025 US6"
documents = engine.search(question)
print(write_answer(question, documents))
print("\nSources:", [d["doc_id"] for d in documents])
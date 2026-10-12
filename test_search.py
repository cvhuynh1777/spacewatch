from app.search import SearchEngine

engine = SearchEngine()

for result in engine.search("Tell me about asteroid 2025 US6"):
    print(round(result["score"], 3), result["document"][:100])

print("-" * 60)

for result in engine.search("asteroids closer than the Moon", filters={"closer_than_moon": True}):
    print(round(result["score"], 3), result["document"][:100])
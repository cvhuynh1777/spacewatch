import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5-5"
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Rules for Claude: stick to the documents, cite them, don't make things up
SYSTEM_PROMPT = """You answer questions about near-Earth asteroids for a service called SpaceWatch.
Use ONLY the documents provided. Each document starts with an ID in square brackets.
Cite the IDs you used in square brackets after the relevant sentence.
If the documents don't contain enough information, say so plainly.
The documents are a sample retrieved from NASA's NeoWs data, not the full dataset,
so never state totals or "the closest ever" unless a document says so.
Keep answers short: 2-5 sentences."""


def write_answer(question, documents):
    """Send the question plus retrieved documents to Claude and return its answer text."""
    if not documents:
        return "I couldn't find any documents matching that question."

    # Label each document with its ID so Claude can cite it
    context = "\n".join(f"[{d['doc_id']}] {d['document']}" for d in documents)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,                                 # room for thinking + the answer
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}],
    )

    # The response may include thinking blocks; keep only the text
    text = "".join(block.text for block in response.content if block.type == "text").strip()

    if response.stop_reason == "max_tokens":
        text += "\n\n[Warning: answer was cut off.]"
    return text
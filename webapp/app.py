import base64
import calendar
import datetime as dt
import html
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="SpaceWatch", layout="wide")

# ---------- Design colors (same as the canvas) ----------
NAVY, PANEL, LINE = "#0D1230", "rgba(22, 29, 69, 0.82)", "#33407A"   # panels are slightly see-through over the stars
CREAM, MUTED = "#F4EAD5", "#B8B2A2"
ORANGE, TEAL = "#F2913D", "#45C4B0"

# ---------- Fonts and custom styles ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=Silkscreen&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.sw-display {{ font-family: 'Syncopate', sans-serif; letter-spacing: 0.06em; color: {CREAM}; margin: 0; }}
.sw-label {{ font-family: 'Silkscreen', monospace; font-size: 12px; color: {MUTED}; }}
.sw-stripes {{ display: flex; gap: 4px; margin: 8px 0 24px; }}
.sw-stripes div {{ flex: 1; height: 6px; border-radius: 3px; }}
.sw-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.sw-tile {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 20px; padding: 20px 22px; display: flex; flex-direction: column; gap: 8px; }}
.sw-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 500; color: {CREAM}; }}
.sw-sub {{ font-size: 13px; color: {MUTED}; }}
.sw-card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 20px; padding: 22px; }}
.sw-badge {{ display: inline-block; padding: 2px 12px; border-radius: 12px; font-size: 12px; border: 1px solid; }}
.sw-facts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 18px 0; }}
.sw-fact b {{ display: block; font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 500; color: {CREAM}; }}
.sw-logo-placeholder {{ width: 240px; height: 240px; margin: 0 auto; border: 2px dashed {TEAL}; border-radius: 24px;
  display: flex; align-items: center; justify-content: center; font-family: 'Silkscreen', monospace; color: {TEAL}; background: {PANEL}; }}
/* ---------- Tactile pill controls ---------- */
[data-testid="stWidgetLabel"] p {{
  font-family: 'Silkscreen', monospace; font-size: 12px; color: {MUTED};
}}
button[data-testid="stBaseButton-pills"],
button[data-testid="stBaseButton-pillsActive"] {{
  min-height: 44px; padding: 8px 18px; border-radius: 999px;
  font-family: 'IBM Plex Sans', sans-serif; font-size: 14px;
  transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.15s ease, border-color 0.15s ease;
}}
/* Unselected: dark raised button with a "ledge" shadow underneath */
button[data-testid="stBaseButton-pills"] {{
  background: {PANEL}; color: {CREAM}; border: 1px solid {LINE};
  box-shadow: 0 3px 0 #070A1F;
}}
button[data-testid="stBaseButton-pills"]:hover {{
  border-color: {TEAL}; transform: translateY(-1px); box-shadow: 0 4px 0 #070A1F;
}}
/* Pressing: the button sinks into its ledge */
button[data-testid="stBaseButton-pills"]:active,
button[data-testid="stBaseButton-pillsActive"]:active {{
  transform: translateY(2px); box-shadow: 0 1px 0 #070A1F;
}}
/* Selected: lit-up orange, like a console button that's switched on */
button[data-testid="stBaseButton-pillsActive"] {{
  background: {ORANGE}; color: {NAVY}; border: 1px solid {ORANGE}; font-weight: 600;
  box-shadow: 0 3px 0 #9C4E14, 0 0 18px rgba(242, 145, 61, 0.35);
}}
</style>
""", unsafe_allow_html=True)

STRIPES = f'<div class="sw-stripes"><div style="background:{ORANGE}"></div><div style="background:{CREAM}"></div><div style="background:{TEAL}"></div></div>'

DATA_START = dt.date(2020, 1, 1)   # the earliest data in the pipeline


# ---------- Data access ----------
cfg = Config()  # the app's own credentials, provided automatically by Databricks


def run_query(query):
    """Run SQL on the warehouse as the app's service principal; return a pandas table."""
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


@st.cache_data(ttl=3600)  # each date range is cached separately for an hour
def load_range(start, end):
    """Every close approach between two dates (inclusive)."""
    # start/end are Python date objects from our own code or the date picker,
    # so .isoformat() always produces a safe 'YYYY-MM-DD' string
    return run_query(f"""
        SELECT neo_id, name, approach_date, miss_distance_km, miss_distance_lunar,
               speed_km_s, diameter_km_min, diameter_km_max, is_hazardous
        FROM spacewatch.gold.neo_approaches
        WHERE approach_date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
    """)


@st.cache_data(ttl=3600)
def load_history(neo_id):
    """Every recorded pass for one asteroid since 2020."""
    if not str(neo_id).isdigit():   # NASA IDs are numbers; refuse anything else before it goes into SQL
        return None
    return run_query(f"""
        SELECT approach_date, miss_distance_km, miss_distance_lunar, speed_km_s
        FROM spacewatch.gold.neo_approaches
        WHERE neo_id = '{neo_id}'
        ORDER BY approach_date
    """)


# ---------- Ask SpaceWatch: search + Claude ----------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # must match the model used to build gold.neo_embeddings
CLAUDE_MODEL = "claude-sonnet-5-5"
ASTEROID_NAME = re.compile(r"\b(\d{4} [A-Z]{2}\d*)\b")   # names like "2025 US6"

SYSTEM_PROMPT = """You answer questions about near-Earth asteroids for a service called SpaceWatch.
Use ONLY the documents provided. Each document starts with an ID in square brackets.
Cite the IDs you used in square brackets after the relevant sentence.
If the documents don't contain enough information, say so plainly.
The documents are a sample retrieved from NASA's NeoWs data, not the full dataset,
so never state totals or "the closest ever" unless a document says so.
Keep answers short: 2-5 sentences."""


PACIFIC = ZoneInfo("America/Los_Angeles")   # "today" means today in Pacific time, not server (UTC) time


def today_pacific():
    return dt.datetime.now(PACIFIC).date()


# Month names and abbreviations -> month number, e.g. "oct" -> 10, "october" -> 10
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTH_DATE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b")


def find_date_range(question):
    """Spot a date in the question and return (start, end), or None if there isn't one.
    Embeddings can't match dates reliably, so dates become an exact filter instead."""
    q = question.lower()
    today = today_pacific()

    if "today" in q or "tonight" in q:
        return today, today
    if "yesterday" in q:
        day = today - dt.timedelta(days=1)
        return day, day
    if "this week" in q or "past week" in q or "last week" in q or "last 7 days" in q:
        return today - dt.timedelta(days=6), today
    if "this month" in q or "last 30 days" in q:
        return today - dt.timedelta(days=29), today

    patterns = [
        (re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", q), "us"),     # 10/11/2026
        (re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q), "iso"),    # 2026-10-11
        (MONTH_DATE.search(q), "words"),                              # October 11, 2026 / oct 11
    ]
    for match, kind in patterns:
        if not match:
            continue
        try:
            if kind == "us":
                day = dt.date(int(match.group(3)), int(match.group(1)), int(match.group(2)))
            elif kind == "iso":
                day = dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            else:
                year = int(match.group(3)) if match.group(3) else today.year
                day = dt.date(year, MONTHS[match.group(1)], int(match.group(2)))
            return day, day
        except ValueError:      # not a real date, like 13/45/2026
            continue
    return None


@st.cache_resource(show_spinner="Loading the asteroid library...")
def load_search_engine():
    """Load all documents + embeddings and the embedding model ONCE, shared by every visitor."""
    docs = run_query("""
        SELECT doc_id, document, approach_date, is_hazardous, closer_than_moon, embedding
        FROM spacewatch.gold.neo_embeddings
    """)
    vectors = np.stack(docs["embedding"].to_numpy()).astype(np.float32)   # 14k x 384 grid
    docs["day"] = pd.to_datetime(docs["approach_date"]).dt.date             # plain dates, for date filters
    model = SentenceTransformer(EMBEDDING_MODEL)
    return docs.drop(columns=["embedding"]), vectors, model


@st.cache_resource
def claude_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def search(question, filters, top_k=8):
    """Hybrid search: exact filters first, then dates and names, then rank by meaning."""
    docs, vectors, model = load_search_engine()
    scores = vectors @ model.encode(question, normalize_embeddings=True)

    # Exact filters: rule out documents that don't match
    for column, value in filters.items():
        scores[(docs[column] != value).to_numpy()] = -np.inf

    # A date in the question ("today", "10/11/2026", "Oct 11"): keep only passes in that range
    date_range = find_date_range(question)
    if date_range:
        start, end = date_range
        in_range = ((docs["day"] >= start) & (docs["day"] <= end)).to_numpy()
        scores[~in_range] = -np.inf
        top_k = 30

    # Named asteroid: keep only its documents, and return all of its passes
    match = ASTEROID_NAME.search(question.upper())
    if match:
        has_name = docs["document"].str.contains(f"({match.group(1)})", regex=False).to_numpy()
        scores[~has_name] = -np.inf
        top_k = 30

    best = [i for i in np.argsort(-scores)[:top_k] if np.isfinite(scores[i])]
    return docs.iloc[best]


# `today` is part of the cache key on purpose: "what passed today?" must not reuse yesterday's answer
@st.cache_data(ttl=86400, show_spinner=False)   # reuse identical answers for a day (saves cost on repeat questions)
def ask(question, filters, today):
    """Retrieve documents, then ask Claude to answer using only those. Returns (answer, sources)."""
    results = search(question, filters)

    date_range = find_date_range(question)
    if results.empty:
        if date_range:
            start, end = date_range
            when = f"{start:%B %d, %Y}" if start == end else f"{start:%b %d} – {end:%b %d, %Y}"
            return (f"There are no close approaches on record for {when} yet. "
                    f"SpaceWatch refreshes from NASA every morning at 6 AM Pacific."), results
        return "I couldn't find any records matching that question and those filters.", results

    # Tell Claude today's date and any date range we filtered on, so it can interpret "today" etc.
    notes = f"Today's date is {today}."
    if date_range:
        notes += f" The documents below were filtered to passes between {date_range[0]} and {date_range[1]}."

    context = "\n".join(f"[{r.doc_id}] {r.document}" for r in results.itertuples())
    response = claude_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{notes}\n\nDocuments:\n{context}\n\nQuestion: {question}"}],
    )
    # The response may contain thinking blocks; keep only the text
    answer = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        answer += "\n\n*(Answer was cut off.)*"
    return answer, results


# ---------- Rate limiting ----------
PER_USER_PER_MINUTE = 5     # questions one person can ask per minute
DAILY_TOTAL = 200           # questions the whole app can answer per day


@st.cache_resource
def rate_limiter():
    """Shared by every visitor (cache_resource = one copy for the whole app)."""
    return {"lock": threading.Lock(), "per_user": defaultdict(deque), "day": None, "count": 0}


def allow_question():
    """Return (True, "") if this person may ask now, else (False, reason)."""
    lim = rate_limiter()
    # Databricks Apps pass the signed-in user's email in this header
    user = st.context.headers.get("X-Forwarded-Email", "anonymous")
    now, today = time.time(), dt.date.today()

    with lim["lock"]:   # stops two visitors updating the counts at the same moment
        if lim["day"] != today:                          # new day: reset the daily count
            lim["day"], lim["count"] = today, 0
        if lim["count"] >= DAILY_TOTAL:
            return False, "SpaceWatch has answered its daily limit of questions. Try again tomorrow."

        recent = lim["per_user"][user]
        while recent and now - recent[0] > 60:           # forget questions older than a minute
            recent.popleft()
        if len(recent) >= PER_USER_PER_MINUTE:
            wait = int(60 - (now - recent[0])) + 1
            return False, f"You're asking quickly! Try again in {wait} seconds."

        recent.append(now)
        lim["count"] += 1
    return True, ""


EXAMPLES = [
    "Which asteroids came closer than the Moon?",
    "Tell me about asteroid 2025 US6",
    "Describe hazardous asteroids that passed recently",
]


def use_example():
    """When an example pill is clicked: copy it into the question box and run it."""
    if st.session_state.example:
        st.session_state.question = st.session_state.example
        st.session_state.run_ask = True


def show_ask():
    st.markdown(STRIPES, unsafe_allow_html=True)
    st.markdown('<h2 class="sw-display" style="font-size:28px">ASK SPACEWATCH</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{MUTED};margin:6px 0 16px">Ask about any asteroid or close approach since 2020. '
                f'Every answer cites the NASA records it came from.</p>', unsafe_allow_html=True)

    st.pills("Try asking", EXAMPLES, selection_mode="single", key="example", on_change=use_example)

    q_col, b_col = st.columns([5, 1], vertical_alignment="bottom")
    question = q_col.text_input("Your question", key="question", max_chars=300,
                                placeholder="e.g. Tell me about asteroid 2025 US6")
    clicked = b_col.button("ASK", type="primary", use_container_width=True)

    ASK_FILTERS = {
        "Closer than the Moon": ":material/dark_mode:",
        "Hazardous only": ":material/warning:",
    }
    chosen = st.pills("Only search", list(ASK_FILTERS), selection_mode="multi", key="ask_filters",
                      format_func=lambda o: f"{ASK_FILTERS[o]} {o}")
    filters = {}
    if "Closer than the Moon" in chosen:
        filters["closer_than_moon"] = True
    if "Hazardous only" in chosen:
        filters["is_hazardous"] = True

    # Run when ASK is clicked, or right after an example pill was picked
    if (clicked or st.session_state.pop("run_ask", False)) and question.strip():
        allowed, reason = allow_question()
        if not allowed:
            st.warning(reason)
        else:
            with st.spinner("Searching NASA records and writing an answer..."):
                try:
                    answer, sources = ask(question.strip(), filters, today_pacific().isoformat())
                    # Keep the result in session_state so it stays visible while you click around the table
                    st.session_state.last_answer = (question.strip(), answer, sources)
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")

    if "last_answer" in st.session_state:
        asked, answer, sources = st.session_state.last_answer
        st.markdown(f'<div class="sw-card" style="margin-top:16px"><span class="sw-label" style="color:{TEAL}">Answer</span>'
                    f'<p style="color:{MUTED};margin:6px 0 0">“{html.escape(asked)}”</p></div>', unsafe_allow_html=True)
        st.markdown(answer)   # plain markdown: Claude's text is shown as text, never as raw HTML
        if len(sources):
            with st.expander(f"Sources ({len(sources)} NASA records)"):
                for r in sources.itertuples():
                    st.markdown(f"`{r.doc_id}` {r.document}")


# ---------- Screen 1: Splash ----------
ASSETS = Path(__file__).parent / "assets"


@st.cache_resource
def logo_video_src():
    """Read assets/logo.mp4 once and turn it into a data URI the page can play directly."""
    path = ASSETS / "logo.mp4"
    if not path.exists():
        return None
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


# ---------- Shared background: starry black sky + old-TV grain (used on BOTH screens) ----------
SKY_CSS = """
<style>
    /* ---------- Black sky with stars (both screens) ---------- */
    .stApp {
      background-color: #000000;
      /* Two layers of tiny stars, tiled at different sizes so the pattern doesn't look repetitive */
      background-image:
        radial-gradient(1px 1px at 25px 35px, rgba(255,255,255,0.85), transparent),
        radial-gradient(1px 1px at 90px 140px, rgba(255,255,255,0.6), transparent),
        radial-gradient(1.5px 1.5px at 160px 70px, rgba(244,234,213,0.9), transparent),
        radial-gradient(1px 1px at 200px 190px, rgba(255,255,255,0.5), transparent),
        radial-gradient(1px 1px at 60px 200px, rgba(69,196,176,0.7), transparent),
        radial-gradient(1px 1px at 120px 20px, rgba(255,255,255,0.7), transparent),
        radial-gradient(1.5px 1.5px at 300px 260px, rgba(255,255,255,0.8), transparent),
        radial-gradient(1px 1px at 40px 310px, rgba(242,145,61,0.6), transparent),
        radial-gradient(1px 1px at 250px 330px, rgba(255,255,255,0.5), transparent);
      background-size: 230px 230px, 230px 230px, 230px 230px, 230px 230px, 230px 230px, 230px 230px,
                       370px 370px, 370px 370px, 370px 370px;
    }
    /* A second, sparser star layer that slowly twinkles */
    .stApp::before {
      content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background-image:
        radial-gradient(1.5px 1.5px at 70px 90px, rgba(255,255,255,0.9), transparent),
        radial-gradient(1px 1px at 220px 40px, rgba(255,255,255,0.8), transparent),
        radial-gradient(1.5px 1.5px at 330px 210px, rgba(244,234,213,0.9), transparent);
      background-size: 410px 410px;
      animation: sw-twinkle 5s ease-in-out infinite;
    }
    @keyframes sw-twinkle { 0%, 100% { opacity: 0.25; } 50% { opacity: 1; } }
    [data-testid="stHeader"] { background: transparent; }

    /* ---------- Old "please stand by" TV look, across the WHOLE screen (kept subtle) ---------- */
    /* Scanlines + a soft dark vignette, painted over the stars but behind the content */
    [data-testid="stAppViewContainer"] {
      position: relative; z-index: 1;
      background:
        repeating-linear-gradient(to bottom, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 3px),
        radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.55) 100%);
      background-attachment: fixed;
    }
    /* Film grain over everything: a noise texture that jitters a few times a second.
       It's larger than the screen (inset: -50%) so the jitter never shows an edge. */
    .stApp::after {
      content: ""; position: fixed; inset: -50%; z-index: 2; pointer-events: none;
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix type='saturate' values='0'/></filter><rect width='160' height='160' filter='url(%23n)'/></svg>");
      opacity: 0.09; mix-blend-mode: screen;
      animation: sw-grain 0.6s steps(6) infinite;
    }

    @keyframes sw-grain {
      0%   { transform: translate(0, 0); }
      20%  { transform: translate(-10%, 5%); }
      40%  { transform: translate(5%, -10%); }
      60%  { transform: translate(-5%, 10%); }
      80%  { transform: translate(10%, -5%); }
      100% { transform: translate(0, 0); }
    }
    /* People who turn off motion in their OS settings get still grain and stars */
    @media (prefers-reduced-motion: reduce) {
      .stApp::after { animation: none; }
      .stApp::before { animation: none; opacity: 0.6; }
    }
</style>
"""


def show_splash():
    st.markdown(SKY_CSS, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* The logo: blended into the sky with a faint old-broadcast flicker */
    .sw-crt { width: min(400px, 90vw); margin: 0 auto; }
    .sw-crt video, .sw-crt img {
      width: 100%; display: block;
      mix-blend-mode: screen;                            /* black parts of the video become see-through, so stars show */
      filter: contrast(1.06) brightness(0.96);          /* slightly washed, like old broadcast */
      animation: sw-flicker 4s infinite;                 /* very faint brightness flicker */
    }
    @keyframes sw-flicker { 0%, 100% { opacity: 1; } 50% { opacity: 0.96; } }
    @media (prefers-reduced-motion: reduce) { .sw-crt video, .sw-crt img { animation: none; } }
    </style>
    """, unsafe_allow_html=True)

    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.write("")
        video_src = logo_video_src()
        if video_src:
            # autoplay + muted + playsinline = plays once like an animated logo, no controls.
            # No "loop": when it finishes, the browser keeps showing the last frame.
            # (Browsers only allow autoplay when the video is muted.)
            # The sw-crt wrapper adds the grain, scanlines and vignette on top of the video
            st.markdown(
                f'<div class="sw-crt"><video src="{video_src}" autoplay muted playsinline '
                f'aria-label="SpaceWatch logo animation"></video></div>',
                unsafe_allow_html=True,
            )
        elif (ASSETS / "logo.gif").exists():
            st.image(str(ASSETS / "logo.gif"), width=400)
        else:
            st.markdown('<div class="sw-logo-placeholder">[YOUR LOGO]</div>', unsafe_allow_html=True)

        st.markdown('<h1 class="sw-display" style="text-align:center;font-size:52px;margin-top:24px">SPACEWATCH</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;color:{MUTED};font-size:18px">Near-Earth asteroid close approaches, from NASA data. Refreshed every morning.</p>', unsafe_allow_html=True)
        st.markdown(STRIPES, unsafe_allow_html=True)

        if st.button("ENTER", type="primary", use_container_width=True):
            st.session_state.entered = True
            st.rerun()


# ---------- Detail panel for one asteroid ----------
def show_details(row):
    name = html.escape(row["name"])
    hazard_color = ORANGE if row["is_hazardous"] else TEAL
    hazard_text = "Potentially hazardous" if row["is_hazardous"] else "Not hazardous"
    jpl_url = f"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={row['neo_id']}"

    st.markdown(f"""
    <div class="sw-card">
      <span class="sw-label">Asteroid</span>
      <h3 class="sw-display" style="font-size:22px;margin:6px 0 10px">{name}</h3>
      <span class="sw-badge" style="color:{hazard_color};border-color:{hazard_color}">{hazard_text}</span>
      <div class="sw-facts">
        <div class="sw-fact"><span class="sw-label">Size</span><b>{row['diameter_km_min']*1000:,.0f}–{row['diameter_km_max']*1000:,.0f} m</b></div>
        <div class="sw-fact"><span class="sw-label">This pass</span><b>{row['approach_date']:%b %d, %Y}</b></div>
        <div class="sw-fact"><span class="sw-label">Miss distance</span><b>{row['miss_distance_km']:,.0f} km</b></div>
        <div class="sw-fact"><span class="sw-label">Moon distances</span><b>{row['miss_distance_lunar']:.2f}</b></div>
        <div class="sw-fact"><span class="sw-label">Speed</span><b>{row['speed_km_s']:.1f} km/s</b></div>
      </div>
      <a href="{jpl_url}" target="_blank" style="color:{ORANGE}">View on NASA JPL →</a>
    </div>
    """, unsafe_allow_html=True)

    history = load_history(row["neo_id"])
    if history is not None and len(history):
        st.markdown(f"**All recorded passes since 2020** ({len(history)})")
        st.dataframe(
            history.assign(
                approach_date=history.approach_date.astype(str),
                miss_distance_km=history.miss_distance_km.map(lambda v: f"{v:,.0f} km"),
                miss_distance_lunar=history.miss_distance_lunar.map(lambda v: f"{v:.2f}"),
                speed_km_s=history.speed_km_s.map(lambda v: f"{v:.1f} km/s"),
            ).rename(columns={"approach_date": "Date", "miss_distance_km": "Miss distance",
                              "miss_distance_lunar": "Moon dist.", "speed_km_s": "Speed"}),
            hide_index=True, use_container_width=True,
        )


# ---------- Screen 2: Dashboard ----------
def show_dashboard():
    st.markdown(SKY_CSS, unsafe_allow_html=True)   # same starry black sky as the splash
    st.markdown('<span class="sw-display" style="font-size:20px">SPACEWATCH</span>', unsafe_allow_html=True)
    st.markdown(STRIPES, unsafe_allow_html=True)

    # --- Date range: quick pills + optional calendar ---
    today = dt.date.today()
    RANGES = {
        "This week": ":material/date_range:",
        "Last 30 days": ":material/calendar_month:",
        "This year": ":material/event:",
        "All time": ":material/all_inclusive:",
        "Custom": ":material/tune:",
    }
    period = st.pills(
        "Date range", list(RANGES), default="This week",
        selection_mode="single", key="period",
        format_func=lambda o: f"{RANGES[o]} {o}",
    ) or "This week"                     # if someone deselects, fall back to this week

    if period == "This week":
        start, end = today - dt.timedelta(days=6), today
    elif period == "Last 30 days":
        start, end = today - dt.timedelta(days=29), today
    elif period == "This year":
        start, end = dt.date(today.year, 1, 1), today
    elif period == "All time":
        start, end = DATA_START, today
    else:  # Custom: show a calendar
        picked = st.date_input(
            "Pick a start and end date",
            value=(today - dt.timedelta(days=6), today),
            min_value=DATA_START, max_value=today, format="MM/DD/YYYY",
        )
        if len(picked) != 2:          # a start date is picked but not an end date yet
            st.info("Pick an end date to update the summary.")
            return
        start, end = picked

    data = load_range(start, end)
    if data.empty:
        st.info("No close approaches recorded in this date range.")
        return

    title = "SUMMARY" if period == "Custom" else period.upper()
    st.markdown(
        f'<h2 class="sw-display" style="font-size:28px">{title}</h2>'
        f'<p class="sw-label" style="margin-top:6px">{start:%b %d, %Y} – {end:%b %d, %Y}</p>',
        unsafe_allow_html=True,
    )

    # --- KPI tiles (computed from the selected range) ---
    closest = data.loc[data.miss_distance_km.idxmin()]
    fastest = data.loc[data.speed_km_s.idxmax()]

    def tile(label, value, sub, color=CREAM):
        return (f'<div class="sw-tile"><span class="sw-label">{label}</span>'
                f'<span class="sw-value" style="color:{color}">{value}</span>'
                f'<span class="sw-sub">{html.escape(str(sub))}</span></div>')

    st.markdown('<div class="sw-tiles">' + "".join([
        tile("Close approaches", f"{len(data):,}", "asteroid passes in this period"),
        tile("Closest pass", f"{closest.miss_distance_km:,.0f} km", closest["name"], TEAL),
        tile("Fastest", f"{fastest.speed_km_s:.1f} km/s", fastest["name"]),
        tile("Potentially hazardous", int(data.is_hazardous.sum()), "flagged by NASA", ORANGE),
    ]) + '</div>', unsafe_allow_html=True)

    # --- Table controls: sort and filters (tactile pills) ---
    SORT_OPTIONS = {
        "Closest first": ("miss_distance_km", True),
        "Fastest first": ("speed_km_s", False),
        "Largest first": ("diameter_km_max", False),
        "Date": ("approach_date", True),
    }
    SORT_ICONS = {
        "Closest first": ":material/my_location:",
        "Fastest first": ":material/speed:",
        "Largest first": ":material/open_in_full:",
        "Date": ":material/calendar_today:",
    }
    FILTER_ICONS = {
        "Hazardous only": ":material/warning:",
        "Closer than the Moon": ":material/dark_mode:",
    }

    c1, c2 = st.columns([3, 2])
    sort_by = c1.pills(
        "Sort by", list(SORT_OPTIONS), default="Closest first",
        selection_mode="single", key="sort",
        format_func=lambda o: f"{SORT_ICONS[o]} {o}",       # add an icon to each label
    ) or "Closest first"                                     # if someone deselects, fall back to closest
    filters = c2.pills(
        "Filter", list(FILTER_ICONS), selection_mode="multi", key="filters",
        format_func=lambda o: f"{FILTER_ICONS[o]} {o}",
    )

    rows = data.copy()
    if "Hazardous only" in filters:
        rows = rows[rows.is_hazardous]
    if "Closer than the Moon" in filters:
        rows = rows[rows.miss_distance_lunar < 1]

    column, ascending = SORT_OPTIONS[sort_by]
    rows = rows.sort_values(column, ascending=ascending).reset_index(drop=True)

    # --- Table (left) + details for the clicked row (right) ---
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown(f"**All approaches** ({len(rows):,}) · click a row for details")
        display = rows.assign(
            approach_date=rows.approach_date.astype(str),
            miss_distance_km=rows.miss_distance_km.map(lambda v: f"{v:,.0f} km"),
            miss_distance_lunar=rows.miss_distance_lunar.map(lambda v: f"{v:.2f}"),
            speed_km_s=rows.speed_km_s.map(lambda v: f"{v:.1f} km/s"),
            size=[f"{a*1000:,.0f}–{b*1000:,.0f} m" for a, b in zip(rows.diameter_km_min, rows.diameter_km_max)],
            is_hazardous=rows.is_hazardous.map({True: "Yes", False: "No"}),
        )[["name", "approach_date", "miss_distance_km", "miss_distance_lunar", "speed_km_s", "size", "is_hazardous"]].rename(columns={
            "name": "Asteroid", "approach_date": "Date", "miss_distance_km": "Miss distance",
            "miss_distance_lunar": "Moon dist.", "speed_km_s": "Speed", "size": "Size", "is_hazardous": "Hazard",
        })

        # The key includes the range, sort and filters, so the selection resets
        # whenever the table's rows change (otherwise a stale row number could point at the wrong asteroid)
        table_key = f"table_{start}_{end}_{sort_by}_{'-'.join(filters)}"

        # on_select="rerun": clicking a row reruns the script with that row selected
        event = st.dataframe(display, hide_index=True, use_container_width=True, height=480,
                             on_select="rerun", selection_mode="single-row", key=table_key)

    with right:
        selected = event.selection.rows
        if selected and selected[0] < len(rows):
            show_details(rows.iloc[selected[0]])
        else:
            st.markdown(f'<div class="sw-card"><span class="sw-label">Details</span>'
                        f'<p style="color:{MUTED};margin:10px 0 0">Click an asteroid in the table to see its details and every pass on record.</p></div>',
                        unsafe_allow_html=True)

    st.write("")
    show_ask()

    st.caption("Data: NASA NeoWs · Pipeline on Databricks · Answers by Claude, grounded with citations · build: v18")


# ---------- Which screen to show ----------
if st.session_state.get("entered"):
    show_dashboard()
else:
    show_splash()
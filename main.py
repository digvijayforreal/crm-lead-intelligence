"""
CRM Lead Intelligence System — FastAPI Backend v2.1
Claude-powered insights with smart caching:
- /leads: scores all leads, calls Claude ONCE per unique profile
- /predict: always calls Claude fresh for manual input
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
import pickle
import os
import re

# ── Anthropic client ───────────────────────────────────────────────────────
_anthropic_client = None
_llm_enabled = False

_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if _api_key:
    try:
        import anthropic
        import httpx
        _anthropic_client = anthropic.Anthropic(
            api_key=_api_key,
            http_client=httpx.Client(),
        )
        _llm_enabled = True
        print("[startup] LLM insights: ENABLED (Claude Haiku)")
    except Exception as e:
        print(f"[startup] LLM disabled: {e}")
else:
    print("[startup] LLM insights: DISABLED (no API key — rule-based fallback active)")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CRM Lead Intelligence API",
    description="XGBoost scoring + Claude-powered insights",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DB_PATH      = os.path.join(BASE_DIR, "leads_db.csv")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print(f"[startup] Model loaded. Industries: {list(label_encoder.classes_)}")

# ── In-memory insight cache (category+industry+engagement bucket as key) ──
_insight_cache: dict[str, str] = {}


def _cache_key(score: float, category: str, industry: str,
               num_calls: int, email_opens: int, website_visits: int) -> str:
    """Coarse engagement bucket so ~100 leads need only ~20 Claude calls."""
    eng = num_calls + email_opens * 0.5 + website_visits * 0.3
    eng_bucket = "low" if eng < 20 else "mid" if eng < 45 else "high"
    return f"{category}|{industry}|{eng_bucket}"


# ── Fallback: rule-based insight ───────────────────────────────────────────
def _rule_based_insight(score, industry, num_calls, email_opens, website_visits):
    if score > 0.8:
        msg = f"High-potential lead from {industry}. Immediate follow-up recommended."
        if website_visits > 35:
            msg += " Strong website engagement signals active buying intent."
        if email_opens > 12:
            msg += " High email responsiveness — personalise the next outreach."
        return msg
    elif score > 0.5:
        msg = f"Moderate potential in {industry}. Nurture with targeted follow-ups."
        if num_calls < 3:
            msg += " Increase call frequency to build rapport."
        else:
            msg += " Consistent engagement — maintain momentum."
        return msg
    else:
        msg = f"Low priority lead from {industry}."
        if email_opens == 0 and website_visits < 5:
            msg += " Very low engagement — consider a re-engagement campaign."
        else:
            msg += " Minimal activity detected — monitor passively."
        return msg


# ── LLM insight via Claude Haiku ──────────────────────────────────────────
def _llm_insight(score, category, industry, num_calls, email_opens, website_visits):
    prompt = f"""You are a senior CRM analyst writing a brief sales insight for a sales rep.

Lead data:
- Industry: {industry}
- Calls made: {num_calls}
- Emails opened: {email_opens}
- Website visits: {website_visits}
- Conversion score: {score:.0%}
- Priority category: {category}

Write a single concise insight (2-3 sentences) that:
1. States the lead potential clearly
2. Explains WHY based on the specific engagement numbers
3. Gives one specific actionable next step

Rules:
- Professional but conversational tone
- No bullet points or headers
- No mention of AI or model or score
- Maximum 55 words
- Begin directly with the insight, no preamble"""

    response = _anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r'^(insight|analysis|summary)\s*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


# ── Insight dispatcher with cache ─────────────────────────────────────────
def generate_insight(score, category, industry, num_calls,
                     email_opens, website_visits, use_cache=True):
    if _llm_enabled and _anthropic_client:
        key = _cache_key(score, category, industry, num_calls, email_opens, website_visits)
        if use_cache and key in _insight_cache:
            return _insight_cache[key]
        try:
            insight = _llm_insight(score, category, industry,
                                   num_calls, email_opens, website_visits)
            if use_cache:
                _insight_cache[key] = insight
            return insight
        except Exception as e:
            print(f"[insight] LLM call failed ({e}), using fallback")
    return _rule_based_insight(score, industry, num_calls, email_opens, website_visits)


# ── Core prediction ────────────────────────────────────────────────────────
def run_prediction(industry, num_calls, email_opens, website_visits, use_cache=True):
    if industry not in label_encoder.classes_:
        industry = "Technology"
    enc      = label_encoder.transform([industry])[0]
    features = np.array([[enc, num_calls, email_opens, website_visits]])
    score    = float(model.predict_proba(features)[0][1])
    category = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    insight  = generate_insight(score, category, industry, num_calls,
                                email_opens, website_visits, use_cache=use_cache)
    return round(score, 4), category, insight


# ── Schemas ────────────────────────────────────────────────────────────────
class LeadOut(BaseModel):
    id: int
    name: str
    company: str
    industry: str
    num_calls: int
    email_opens: int
    website_visits: int
    score: float
    category: Literal["High", "Medium", "Low"]
    insight: str

class PredictRequest(BaseModel):
    name: str = "Unknown"
    company: str = ""
    industry: str
    num_calls: int
    email_opens: int
    website_visits: int

class PredictResponse(BaseModel):
    name: str
    company: str
    industry: str
    score: float
    category: Literal["High", "Medium", "Low"]
    insight: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "XGBoost",
        "version": "2.1.0",
        "llm_insights": _llm_enabled,
        "cached_insights": len(_insight_cache),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/leads", response_model=list[LeadOut])
def get_leads():
    """
    Auto pipeline — scores all leads.
    Claude is called once per unique engagement profile (cached),
    so 100 leads may only need ~8-12 actual API calls.
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="leads_db.csv not found")
    df = pd.read_csv(DB_PATH)
    results = []
    for _, row in df.iterrows():
        score, category, insight = run_prediction(
            str(row["industry"]), int(row["num_calls"]),
            int(row["email_opens"]), int(row["website_visits"]),
            use_cache=True,
        )
        results.append(LeadOut(
            id=int(row["id"]), name=str(row["name"]), company=str(row["company"]),
            industry=str(row["industry"]), num_calls=int(row["num_calls"]),
            email_opens=int(row["email_opens"]), website_visits=int(row["website_visits"]),
            score=score, category=category, insight=insight))
    print(f"[leads] Served {len(results)} leads. Cache size: {len(_insight_cache)}")
    return results

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Manual pipeline — always calls Claude fresh (no cache)."""
    score, category, insight = run_prediction(
        req.industry, req.num_calls, req.email_opens, req.website_visits,
        use_cache=False,
    )
    return PredictResponse(
        name=req.name, company=req.company, industry=req.industry,
        score=score, category=category, insight=insight)

@app.get("/industries")
def get_industries():
    return {"industries": list(label_encoder.classes_)}

"""
CRM Lead Intelligence System — FastAPI Backend
LLM-upgraded: Claude (Anthropic) powers the insight engine
Fallback: rule-based logic if API key missing or call fails
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
    print("[startup] LLM insights: DISABLED (no ANTHROPIC_API_KEY — using rule-based fallback)")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CRM Lead Intelligence API",
    description="XGBoost scoring + Claude-powered insights",
    version="2.0.0",
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

Write a single concise insight (2-3 sentences max) that:
1. States the lead's potential clearly
2. Explains WHY based on the engagement data
3. Gives one specific, actionable next step

Rules:
- Professional but human tone
- No bullet points, no headers
- No mention of AI or model
- Under 60 words
- Start directly with the insight, no preamble"""

    response = _anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r'^(insight|analysis|summary)\s*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


# ── Insight dispatcher ─────────────────────────────────────────────────────
def generate_insight(score, category, industry, num_calls, email_opens, website_visits):
    if _llm_enabled and _anthropic_client:
        try:
            return _llm_insight(score, category, industry, num_calls, email_opens, website_visits)
        except Exception as e:
            print(f"[insight] LLM call failed ({e}), using fallback")
    return _rule_based_insight(score, industry, num_calls, email_opens, website_visits)


# ── Core prediction ────────────────────────────────────────────────────────
def run_prediction(industry, num_calls, email_opens, website_visits):
    if industry not in label_encoder.classes_:
        industry = "Technology"
    enc      = label_encoder.transform([industry])[0]
    features = np.array([[enc, num_calls, email_opens, website_visits]])
    score    = float(model.predict_proba(features)[0][1])
    category = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    insight  = generate_insight(score, category, industry, num_calls, email_opens, website_visits)
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
        "version": "2.0.0",
        "llm_insights": _llm_enabled,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/leads", response_model=list[LeadOut])
def get_leads():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="leads_db.csv not found")
    df = pd.read_csv(DB_PATH)
    results = []
    for _, row in df.iterrows():
        score, category, insight = run_prediction(
            str(row["industry"]), int(row["num_calls"]),
            int(row["email_opens"]), int(row["website_visits"]))
        results.append(LeadOut(
            id=int(row["id"]), name=str(row["name"]), company=str(row["company"]),
            industry=str(row["industry"]), num_calls=int(row["num_calls"]),
            email_opens=int(row["email_opens"]), website_visits=int(row["website_visits"]),
            score=score, category=category, insight=insight))
    return results

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    score, category, insight = run_prediction(
        req.industry, req.num_calls, req.email_opens, req.website_visits)
    return PredictResponse(
        name=req.name, company=req.company, industry=req.industry,
        score=score, category=category, insight=insight)

@app.get("/industries")
def get_industries():
    return {"industries": list(label_encoder.classes_)}

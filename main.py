from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(title="CRM Lead Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ─────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

DB_PATH = os.path.join(BASE, "leads_db.csv")
print("Model loaded. Industries:", list(label_encoder.classes_))


# ── GenAI Insight Engine ──────────────────────────────────
def generate_insight(score, industry, num_calls, email_opens, website_visits):
    if score > 0.8:
        msg = f"High-potential lead from {industry}. Immediate follow-up recommended."
        if website_visits > 35:
            msg += " Strong website engagement signals buying intent."
        if email_opens > 12:
            msg += " High email responsiveness — personalise your outreach."
        return msg
    elif score > 0.5:
        msg = f"Moderate potential in {industry}. Nurture with targeted follow-ups."
        if num_calls < 3:
            msg += " Increase call frequency to build rapport."
        else:
            msg += " Consistent engagement — keep the momentum."
        return msg
    else:
        msg = f"Low priority lead from {industry}."
        if email_opens == 0 and website_visits < 5:
            msg += " Very low engagement — consider a re-engagement campaign."
        else:
            msg += " Minimal activity — monitor passively."
        return msg


# ── Core prediction logic ─────────────────────────────────
def run_prediction(industry, num_calls, email_opens, website_visits):
    if industry not in label_encoder.classes_:
        industry = "Technology"
    enc      = label_encoder.transform([industry])[0]
    features = np.array([[enc, num_calls, email_opens, website_visits]])
    score    = float(model.predict_proba(features)[0][1])
    category = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    insight  = generate_insight(score, industry, num_calls, email_opens, website_visits)
    return round(score, 4), category, insight


# ── Schemas ───────────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "model": "XGBoost"}

@app.get("/leads", response_model=list[LeadOut])
def get_leads():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="leads_db.csv not found")
    df = pd.read_csv(DB_PATH)
    results = []
    for _, row in df.iterrows():
        score, category, insight = run_prediction(
            str(row["industry"]), int(row["num_calls"]),
            int(row["email_opens"]), int(row["website_visits"])
        )
        results.append(LeadOut(
            id=int(row["id"]), name=str(row["name"]), company=str(row["company"]),
            industry=str(row["industry"]), num_calls=int(row["num_calls"]),
            email_opens=int(row["email_opens"]), website_visits=int(row["website_visits"]),
            score=score, category=category, insight=insight
        ))
    return results

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    score, category, insight = run_prediction(
        req.industry, req.num_calls, req.email_opens, req.website_visits
    )
    return PredictResponse(
        name=req.name, company=req.company, industry=req.industry,
        score=score, category=category, insight=insight
    )

@app.get("/industries")
def industries():
    return {"industries": list(label_encoder.classes_)}
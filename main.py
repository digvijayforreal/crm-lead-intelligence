"""
CRM Lead Intelligence System — FastAPI Backend v3.0
Supports user-uploaded datasets with flexible column mapping.
Existing /leads and /predict endpoints unchanged.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import numpy as np
import pickle
import os
import re
import io

# ── Anthropic (optional) ───────────────────────────────────────────────────
_anthropic_client = None
_llm_enabled = False
_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if _api_key:
    try:
        import anthropic, httpx
        _anthropic_client = anthropic.Anthropic(api_key=_api_key, http_client=httpx.Client())
        _llm_enabled = True
        print("[startup] LLM insights: ENABLED (Claude Haiku)")
    except Exception as e:
        print(f"[startup] LLM disabled: {e}")
else:
    print("[startup] LLM insights: DISABLED — rule-based fallback active")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="CRM Lead Intelligence API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH    = os.path.join(BASE_DIR, "label_encoder.pkl")
DEFAULT_DB      = os.path.join(BASE_DIR, "leads_db.csv")
UPLOADED_RAW    = os.path.join(BASE_DIR, "uploaded_raw.csv")
UPLOADED_MAPPED = os.path.join(BASE_DIR, "uploaded_mapped.csv")

# ── Model state (mutable — hot-reloaded after retraining) ─────────────────
_state = {}

def _load_model():
    from sklearn.preprocessing import LabelEncoder
    with open(MODEL_PATH, "rb") as f:
        _state["model"] = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        _state["label_encoder"] = pickle.load(f)
    print(f"[model] Loaded. Industries: {list(_state['label_encoder'].classes_)}")

_load_model()

# ── Insight cache & circuit ────────────────────────────────────────────────
_insight_cache: dict[str, str] = {}
_llm_circuit_open = False

def _cache_key(category, industry, num_calls, email_opens, website_visits):
    eng = num_calls + email_opens * 0.5 + website_visits * 0.3
    bucket = "low" if eng < 20 else "mid" if eng < 45 else "high"
    return f"{category}|{industry}|{bucket}"

def _rule_based_insight(score, industry, num_calls, email_opens, website_visits):
    if score > 0.8:
        msg = f"High-potential lead from {industry}. Immediate follow-up recommended."
        if website_visits > 35: msg += " Strong website engagement signals active buying intent."
        if email_opens > 12:    msg += " High email responsiveness — personalise the next outreach."
        return msg
    elif score > 0.5:
        msg = f"Moderate potential in {industry}. Nurture with targeted follow-ups."
        msg += " Increase call frequency to build rapport." if num_calls < 3 else " Consistent engagement — maintain momentum."
        return msg
    else:
        msg = f"Low priority lead from {industry}."
        msg += " Very low engagement — consider a re-engagement campaign." if email_opens == 0 and website_visits < 5 else " Minimal activity detected — monitor passively."
        return msg

def _llm_insight(score, category, industry, num_calls, email_opens, website_visits):
    prompt = f"""You are a senior CRM analyst writing a brief sales insight for a sales rep.
Lead: Industry={industry}, Calls={num_calls}, Emails={email_opens}, Visits={website_visits}, Score={score:.0%}, Category={category}
Write 2-3 sentences: state potential, explain why from the numbers, give one action.
Rules: professional tone, no bullets, no mention of AI, max 55 words, start directly."""
    r = _anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=120,
        messages=[{"role": "user", "content": prompt}]
    )
    return re.sub(r'^(insight|analysis)\s*:\s*', '', r.content[0].text.strip(), flags=re.IGNORECASE).strip()

def generate_insight(score, category, industry, num_calls, email_opens, website_visits, use_cache=True):
    global _llm_circuit_open
    if _llm_enabled and _anthropic_client and not _llm_circuit_open:
        key = _cache_key(category, industry, num_calls, email_opens, website_visits)
        if use_cache and key in _insight_cache:
            return _insight_cache[key]
        try:
            ins = _llm_insight(score, category, industry, num_calls, email_opens, website_visits)
            if use_cache: _insight_cache[key] = ins
            return ins
        except Exception as e:
            err = str(e)
            print(f"[insight] LLM failed ({err[:80]}), using fallback")
            if "credit balance" in err or "insufficient_quota" in err:
                _llm_circuit_open = True
    return _rule_based_insight(score, industry, num_calls, email_opens, website_visits)

# ── Core prediction ────────────────────────────────────────────────────────
def run_prediction(industry, num_calls, email_opens, website_visits, use_cache=True):
    le = _state["label_encoder"]
    m  = _state["model"]
    if industry not in le.classes_:
        industry = le.classes_[0]
    enc      = le.transform([industry])[0]
    features = np.array([[enc, num_calls, email_opens, website_visits]])
    score    = float(m.predict_proba(features)[0][1])
    category = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    insight  = generate_insight(score, category, industry, num_calls, email_opens, website_visits, use_cache)
    return round(score, 4), category, insight

# ── Active dataset helper ──────────────────────────────────────────────────
def _active_db() -> str:
    """Return mapped dataset path if available, else default."""
    return UPLOADED_MAPPED if os.path.exists(UPLOADED_MAPPED) else DEFAULT_DB

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

class MappingRequest(BaseModel):
    mapping: dict  # e.g. {"industry": "col_a", "num_calls": "col_b", ...}

# ── Endpoints: existing (unchanged) ───────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running", "version": "3.0.0",
        "active_dataset": "uploaded" if os.path.exists(UPLOADED_MAPPED) else "default",
        "llm_insights": _llm_enabled,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/leads", response_model=list[LeadOut])
def get_leads():
    db = _active_db()
    if not os.path.exists(db):
        raise HTTPException(status_code=500, detail="No dataset found")
    df = pd.read_csv(db)
    results = []
    for i, row in df.iterrows():
        name    = str(row.get("name", f"Lead {i+1}"))
        company = str(row.get("company", "Unknown"))
        industry = str(row.get("industry", "Technology"))
        calls   = int(row.get("num_calls", 0))
        emails  = int(row.get("email_opens", 0))
        visits  = int(row.get("website_visits", 0))
        score, category, insight = run_prediction(industry, calls, emails, visits, use_cache=True)
        results.append(LeadOut(
            id=int(row.get("id", i+1)), name=name, company=company,
            industry=industry, num_calls=calls, email_opens=emails,
            website_visits=visits, score=score, category=category, insight=insight))
    return results

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    score, category, insight = run_prediction(req.industry, req.num_calls, req.email_opens, req.website_visits, use_cache=False)
    return PredictResponse(name=req.name, company=req.company, industry=req.industry, score=score, category=category, insight=insight)

@app.get("/industries")
def get_industries():
    return {"industries": list(_state["label_encoder"].classes_)}

# ── NEW: POST /upload-dataset ──────────────────────────────────────────────
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept a CSV file, save it, return its column names.
    Does NOT validate — just reads and returns what it finds.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="CSV appears to be empty")

    # Save raw upload
    df.to_csv(UPLOADED_RAW, index=False)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "sample": df.head(3).to_dict(orient="records"),
    }

# ── NEW: POST /apply-mapping ───────────────────────────────────────────────
@app.post("/apply-mapping")
def apply_mapping(req: MappingRequest):
    """
    Apply user-defined column mapping to the uploaded dataset.
    Mapping keys: industry, num_calls, email_opens, website_visits, converted
    Mapping values: actual column names in the uploaded CSV
    Missing columns are filled with sensible defaults.
    """
    if not os.path.exists(UPLOADED_RAW):
        raise HTTPException(status_code=400, detail="No uploaded dataset found. Upload a CSV first.")

    df = pd.read_csv(UPLOADED_RAW)
    mapping = req.mapping  # e.g. {"industry": "sector", "num_calls": "calls_made"}

    REQUIRED = ["industry", "num_calls", "email_opens", "website_visits"]
    OPTIONAL  = ["converted", "name", "company", "id"]
    DEFAULTS  = {"industry": "Unknown", "num_calls": 0, "email_opens": 0, "website_visits": 0, "converted": 0}

    out = pd.DataFrame()
    warnings = []

    # Map or default each required feature
    for std_col in REQUIRED + OPTIONAL:
        src_col = mapping.get(std_col)
        if src_col and src_col in df.columns:
            out[std_col] = df[src_col]
        elif src_col and src_col not in df.columns:
            warnings.append(f"Mapped column '{src_col}' not found for '{std_col}' — using default")
            out[std_col] = DEFAULTS.get(std_col, "")
        else:
            # Not mapped — use default
            if std_col in REQUIRED:
                warnings.append(f"'{std_col}' not mapped — filled with default ({DEFAULTS[std_col]})")
            if std_col == "id":
                out["id"] = range(1, len(df) + 1)
            elif std_col == "name":
                out["name"] = [f"Lead {i+1}" for i in range(len(df))]
            elif std_col == "company":
                out["company"] = "Unknown"
            else:
                out[std_col] = DEFAULTS.get(std_col, 0)

    # Clean numeric columns
    for col in ["num_calls", "email_opens", "website_visits", "converted"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # Fill industry nulls
    out["industry"] = out["industry"].fillna("Unknown").astype(str)

    # Save mapped dataset
    out.to_csv(UPLOADED_MAPPED, index=False)

    return {
        "status": "mapping_applied",
        "rows": len(out),
        "columns": list(out.columns),
        "warnings": warnings,
        "sample": out.head(3).to_dict(orient="records"),
    }

# ── NEW: POST /retrain ─────────────────────────────────────────────────────
@app.post("/retrain")
def retrain():
    """
    Retrain XGBoost on the mapped dataset.
    Hot-reloads model into memory — no server restart needed.
    """
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    if not os.path.exists(UPLOADED_MAPPED):
        raise HTTPException(status_code=400, detail="No mapped dataset. Apply mapping first.")

    df = pd.read_csv(UPLOADED_MAPPED)

    if "converted" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset has no 'converted' column for training.")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail=f"Need at least 20 rows to train. Got {len(df)}.")

    if df["converted"].nunique() < 2:
        raise HTTPException(status_code=400, detail="Target column 'converted' must have both 0 and 1 values.")

    # Encode industry
    le = LabelEncoder()
    df["industry_enc"] = le.fit_transform(df["industry"].astype(str))

    X = df[["industry_enc", "num_calls", "email_opens", "website_visits"]]
    y = df["converted"]

    # Train/test split
    test_size = 0.2 if len(df) >= 50 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() >= 2 else None
    )

    new_model = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0
    )
    new_model.fit(X_train, y_train)

    # Evaluate
    y_prob = new_model.predict_proba(X_test)[:, 1]
    try:
        auc = round(roc_auc_score(y_test, y_prob), 4)
    except Exception:
        auc = None

    # Save model and encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(new_model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # Hot-reload into live state
    _state["model"] = new_model
    _state["label_encoder"] = le

    # Clear insight cache (insights were for old model)
    _insight_cache.clear()

    print(f"[retrain] Model retrained. AUC={auc}. Industries: {list(le.classes_)}")

    return {
        "status": "retrained",
        "rows_used": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "roc_auc": auc,
        "industries": list(le.classes_),
    }

# ── NEW: POST /reset-dataset ───────────────────────────────────────────────
@app.post("/reset-dataset")
def reset_dataset():
    """Remove uploaded dataset and revert to default leads_db.csv."""
    removed = []
    for p in [UPLOADED_RAW, UPLOADED_MAPPED]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.basename(p))
    # Reload original model
    _load_model()
    _insight_cache.clear()
    return {"status": "reset", "removed": removed, "active_dataset": "default"}

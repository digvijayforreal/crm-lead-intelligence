from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import numpy as np
import pickle, os, io, httpx

app = FastAPI(title="CRM Lead Intelligence API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────
BASE              = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB        = os.path.join(BASE, "leads_db.csv")
UPLOADED_DB       = os.path.join(BASE, "uploaded_dataset.csv")
MAPPED_DB         = os.path.join(BASE, "mapped_dataset.csv")
MODEL_PATH        = os.path.join(BASE, "model.pkl")
ENCODER_PATH      = os.path.join(BASE, "label_encoder.pkl")

# ── Config flags ──────────────────────────────────────────
# MODE: "groq" | "rule"  (Claude kept as optional upgrade)
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama3-8b-8192"          # free, fast, high quality
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODE      = "groq" if GROQ_API_KEY else "rule"

# ── Load model ────────────────────────────────────────────
def load_model():
    global model, label_encoder
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

def retrain_from_default():
    """Retrain model from leads_db.csv using current XGBoost version."""
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    print("[startup] Retraining model with current XGBoost version...")
    df = pd.read_csv(DEFAULT_DB)
    le = LabelEncoder()
    df["industry_enc"] = le.fit_transform(df["industry"].astype(str))
    X = df[["industry_enc", "num_calls", "email_opens", "website_visits"]].values
    y = df["converted"].values
    clf = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss")
    clf.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print("[startup] Model retrained and saved successfully.")

try:
    load_model()
    # Quick sanity check — trigger the exact call that was failing
    import numpy as _np
    _test = model.predict_proba(_np.array([[0, 5, 10, 30]]))[0][1]
    print(f"[startup] Model sanity check passed (score={_test:.3f})")
except Exception as e:
    print(f"[startup] Model load/check failed ({e}) — retraining from default dataset...")
    retrain_from_default()
    load_model()
    print("[startup] Model reloaded after retraining.")

print(f"[startup] LLM mode: {LLM_MODE.upper()}")
if LLM_MODE == "groq":
    print(f"[startup] Groq model: {GROQ_MODEL}")
print(f"[startup] Model loaded. Industries: {list(label_encoder.classes_)}")

# ── Insight cache ─────────────────────────────────────────
_insight_cache: dict = {}

def _cache_key(score: float, industry: str, num_calls: int,
               email_opens: int, website_visits: int) -> str:
    cat = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    eng = "high" if (num_calls > 6 and email_opens > 10 and website_visits > 30) \
          else "low" if (num_calls < 3 and email_opens < 5 and website_visits < 10) \
          else "mid"
    return f"{cat}|{industry}|{eng}"

# ── Rule-based fallback ───────────────────────────────────
def rule_based_insight(score: float, industry: str, num_calls: int,
                       email_opens: int, website_visits: int) -> str:
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

# ── Groq LLM insight ──────────────────────────────────────
def groq_insight(score: float, industry: str, num_calls: int,
                 email_opens: int, website_visits: int) -> str:
    category = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    prompt = (
        f"You are a concise B2B sales analyst. A CRM lead has these details:\n"
        f"Industry: {industry}\n"
        f"Calls made: {num_calls}\n"
        f"Email opens: {email_opens}\n"
        f"Website visits: {website_visits}\n"
        f"Conversion score: {score:.2f} ({category} priority)\n\n"
        f"Write a 2-sentence actionable recommendation for the sales rep. "
        f"Be specific to the numbers. No generic advice."
    )
    api_key = GROQ_API_KEY.strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "temperature": 0.5,
    }
    resp = httpx.post(GROQ_BASE_URL, json=body, headers=headers, timeout=8.0)
    if not resp.is_success:
        print(f"[groq] HTTP {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ── Main insight dispatcher ───────────────────────────────
def generate_insight(score: float, industry: str, num_calls: int,
                     email_opens: int, website_visits: int) -> str:
    key = _cache_key(score, industry, num_calls, email_opens, website_visits)
    if key in _insight_cache:
        return _insight_cache[key]

    insight = None
    if LLM_MODE == "groq" and GROQ_API_KEY:
        try:
            insight = groq_insight(score, industry, num_calls, email_opens, website_visits)
        except Exception as e:
            print(f"[insight] Groq failed ({e}), using rule-based fallback")

    if insight is None:
        insight = rule_based_insight(score, industry, num_calls, email_opens, website_visits)

    _insight_cache[key] = insight
    return insight

# ── Core prediction ───────────────────────────────────────
def run_prediction(industry: str, num_calls: int,
                   email_opens: int, website_visits: int):
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

class MappingRequest(BaseModel):
    mapping: dict
    fill_defaults: bool = True

# ── Active dataset helper ─────────────────────────────────
def active_db_path() -> str:
    if os.path.exists(MAPPED_DB):
        return MAPPED_DB
    if os.path.exists(UPLOADED_DB):
        return UPLOADED_DB
    return DEFAULT_DB

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "XGBoost",
        "version": "4.0.0",
        "llm_mode": LLM_MODE,
        "llm_model": GROQ_MODEL if LLM_MODE == "groq" else "rule-based",
        "cached_insights": len(_insight_cache),
        "active_dataset": "custom" if os.path.exists(MAPPED_DB) else "default",
    }

@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": LLM_MODE}

@app.get("/leads", response_model=list[LeadOut])
def get_leads():
    db_path = active_db_path()
    if not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="Dataset not found")
    df = pd.read_csv(db_path)

    # Sanitise string columns — replace NaN/None with fallback strings
    df["name"]    = df["name"].fillna("Unknown").astype(str).str.strip()
    df["company"] = df["company"].fillna("Unknown").astype(str).str.strip()
    df["industry"]= df["industry"].fillna("Technology").astype(str).str.strip()

    # Replace empty strings that came through as NaN after strip
    df["name"]    = df["name"].replace({"": "Unknown", "nan": "Unknown"})
    df["company"] = df["company"].replace({"": "Unknown", "nan": "Unknown"})

    results = []
    for _, row in df.iterrows():
        score, category, insight = run_prediction(
            str(row["industry"]), int(row["num_calls"]),
            int(row["email_opens"]), int(row["website_visits"])
        )
        results.append(LeadOut(
            id=int(row["id"]),
            name=str(row["name"]),
            company=str(row["company"]),
            industry=str(row["industry"]),
            num_calls=int(row["num_calls"]),
            email_opens=int(row["email_opens"]),
            website_visits=int(row["website_visits"]),
            score=score, category=category, insight=insight,
        ))
    print(f"[leads] Served {len(results)} leads. Cache size: {len(_insight_cache)}")
    return results

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    score, category, _ = run_prediction(
        req.industry, req.num_calls, req.email_opens, req.website_visits
    )
    # Fresh insight for manual predictions — bypass cache
    insight = None
    if LLM_MODE == "groq" and GROQ_API_KEY:
        try:
            insight = groq_insight(score, req.industry, req.num_calls,
                                   req.email_opens, req.website_visits)
        except Exception as e:
            print(f"[predict] Groq failed ({e}), using rule-based")
    if insight is None:
        insight = rule_based_insight(score, req.industry, req.num_calls,
                                     req.email_opens, req.website_visits)
    return PredictResponse(
        name=req.name, company=req.company, industry=req.industry,
        score=score, category=category, insight=insight,
    )

@app.get("/industries")
def industries():
    return {"industries": list(label_encoder.classes_)}

@app.post("/clear-cache")
def clear_cache():
    count = len(_insight_cache)
    _insight_cache.clear()
    return {"message": f"Cleared {count} cached insights. Next /leads call will regenerate via Groq."}

@app.get("/test-groq")
def test_groq():
    """Test Groq connectivity and return detailed diagnostics."""
    key = GROQ_API_KEY.strip()
    if not key:
        return {"status": "error", "reason": "GROQ_API_KEY env var is not set"}
    key_preview = key[:8] + "..." + key[-4:] if len(key) > 12 else "too_short"
    try:
        resp = httpx.post(
            GROQ_BASE_URL,
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=8.0,
        )
        if resp.is_success:
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            return {"status": "ok", "model": GROQ_MODEL, "reply": reply, "key_preview": key_preview}
        else:
            return {"status": "error", "http_status": resp.status_code,
                    "body": resp.text[:400], "key_preview": key_preview}
    except Exception as e:
        return {"status": "exception", "error": str(e), "key_preview": key_preview}

# ── Dataset upload & mapping (preserved from v3) ──────────
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df.to_csv(UPLOADED_DB, index=False)
    return {
        "message": "Dataset uploaded successfully",
        "rows": len(df),
        "columns": list(df.columns),
        "sample": df.head(3).to_dict(orient="records"),
    }

@app.post("/apply-mapping")
def apply_mapping(req: MappingRequest):
    if not os.path.exists(UPLOADED_DB):
        raise HTTPException(status_code=400, detail="No uploaded dataset found")
    df = pd.read_csv(UPLOADED_DB)
    mapped = pd.DataFrame()
    required = ["id", "name", "company", "industry",
                "num_calls", "email_opens", "website_visits", "converted"]
    warnings = []
    for std_col in required:
        src = req.mapping.get(std_col)
        if src and src in df.columns:
            mapped[std_col] = df[src]
        elif src == "__default__" or not src:
            if req.fill_defaults:
                mapped[std_col] = 0 if std_col not in ("id", "name", "company", "industry") else ""
                warnings.append(f"{std_col} not mapped — filled with default")
            else:
                raise HTTPException(status_code=400, detail=f"Required column {std_col} not mapped")
        else:
            mapped[std_col] = df[src] if src in df.columns else 0
            warnings.append(f"{std_col} source column {src} not found — using 0")
    if "id" not in mapped.columns or mapped["id"].isnull().all():
        mapped["id"] = range(1, len(mapped) + 1)
    mapped.to_csv(MAPPED_DB, index=False)
    _insight_cache.clear()
    return {"message": "Mapping applied", "rows": len(mapped), "warnings": warnings}

@app.post("/retrain")
def retrain():
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    if not os.path.exists(MAPPED_DB):
        raise HTTPException(status_code=400, detail="No mapped dataset found")
    df = pd.read_csv(MAPPED_DB)
    le = LabelEncoder()
    df["industry_enc"] = le.fit_transform(df["industry"].astype(str))
    X = df[["industry_enc", "num_calls", "email_opens", "website_visits"]].values
    y = df["converted"].values
    clf = XGBClassifier(n_estimators=100, max_depth=4,
                        use_label_encoder=False, eval_metric="logloss")
    clf.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    load_model()
    _insight_cache.clear()
    return {"message": "Model retrained successfully", "rows": len(df),
            "industries": list(le.classes_)}

@app.post("/reset-dataset")
def reset_dataset():
    for path in [MAPPED_DB, UPLOADED_DB]:
        if os.path.exists(path):
            os.remove(path)
    _insight_cache.clear()
    return {"message": "Reset to default dataset"}

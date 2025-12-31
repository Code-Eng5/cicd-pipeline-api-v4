from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ---------------- Load artifacts ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "deployment_artifacts")

preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocess.pkl"))
model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm.pkl"))

job_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "job_freq_map.pkl"))
stage_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "stage_freq_map.pkl"))
branch_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "branch_freq_map.pkl"))
task_freq_map = joblib.load(os.path.join(ARTIFACTS_DIR, "task_freq_map.pkl"))

# ---------------- FastAPI setup ----------------
app = FastAPI(title="CI/CD Pipeline Prediction API")

# ---------------- Input schema ----------------
class PipelineInput(BaseModel):
    job_name: str = "unknown"
    stage_name: str = "unknown"
    branch: str = "unknown"
    environment: str = "unknown"
    user: str = "unknown"
    task_name: str = "unknown"

# ---------------- Health endpoint ----------------
@app.get("/")
def health():
    return {"status": "running"}

# ---------------- Predict endpoint ----------------
@app.post("/predict")
def predict(data: PipelineInput):
    # 1️⃣ Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # 2️⃣ Fill missing values for safety
    df = df.fillna("unknown")

    # 3️⃣ Add frequency features exactly as in training
    df['job_freq'] = df['job_name'].map(job_freq_map).fillna(1)
    df['stage_freq'] = df['stage_name'].map(stage_freq_map).fillna(1)
    df['branch_freq'] = df['branch'].map(branch_freq_map).fillna(1)
    df['task_freq'] = df['task_name'].map(task_freq_map).fillna(1)

    # 4️⃣ Ensure all preprocessor columns exist
    for col in preprocessor.feature_names_in_:
        if col not in df.columns:
            df[col] = "unknown" if col in df.select_dtypes(include='object').columns else 0

    # 5️⃣ Reorder columns to match preprocessor
    df = df[preprocessor.feature_names_in_]

    # 6️⃣ Transform and predict
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]
    prediction = "Failure" if prob >= 0.5 else "Success"

    return {"prediction": prediction, "confidence": round(float(prob), 3)}

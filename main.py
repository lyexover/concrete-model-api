import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialisation de l'application
app = FastAPI(title="Concrete Strength API")

# 2. Configuration du CORS (Indispensable pour la connexion avec Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (Vercel, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise POST, GET, etc.
    allow_headers=["*"],  # Autorise tous les headers
)

# 3. Chargement du modèle et du scaler
# Assure-toi que ces fichiers sont à la racine de ton dossier sur GitHub
try:
    model = joblib.load("concrete_model.joblib")
    scaler = joblib.load("reg_scaler.joblib")
    print("✅ Modèle et Scaler chargés avec succès.")
except Exception as e:
    print(f"❌ Erreur de chargement des artefacts : {e}")
    model = None
    scaler = None

# 4. Schéma des données d'entrée (Doit correspondre à tes 8 variables)
class ConcreteInput(BaseModel):
    cement: float
    slag: float
    ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age: float

@app.get("/")
def health_check():
    return {"status": "online", "message": "API is running"}

@app.post("/predict")
def predict_strength(data: ConcreteInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé sur le serveur.")

    try:
        # L'ordre ici doit être EXACTEMENT celui utilisé lors de l'entraînement
        # (D'après ton notebook de preprocessing)
        features = np.array([[
            data.cement,
            data.slag,
            data.ash,
            data.water,
            data.superplasticizer,
            data.coarse_aggregate,
            data.fine_aggregate,
            data.age
        ]])

        # Appliquer le scaling (Crucial car le modèle a été entraîné sur des données scalées)
        features_scaled = scaler.transform(features)

        # Prédiction
        prediction = model.predict(features_scaled)

        return {
            "prediction": float(prediction[0]),
            "unit": "MPa"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
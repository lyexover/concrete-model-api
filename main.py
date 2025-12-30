# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialisation de l'app
app = FastAPI(title="Concrete Strength API", version="1.0")

# --- Configuration CORS (Pour autoriser ton frontend Next.js) ---
# On autorise tout pour le développement, mais idéalement, mets l'URL de ton frontend Vercel plus tard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement des artefacts (Modèle et Scaler) ---
# On utilise des chemins relatifs pour que ça marche en local et sur le cloud
try:
    model = joblib.load("concrete_model.joblib")
    # Si tu n'as pas retrouvé ton scaler, commente la ligne suivante et les lignes de transformation plus bas
    # MAIS tes prédictions seront fausses sans le scaler.
    scaler = joblib.load("reg_scaler.joblib") 
    print("✅ Modèle et Scaler chargés avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle/scaler : {e}")
    model = None
    scaler = None

# --- Schéma des données (Pydantic) ---
# Remplace les noms ci-dessous par les noms EXACTS de tes colonnes d'entraînement
# Je mets ici les noms standards du dataset "Concrete", adapte-les si besoin.
class ConcreteInput(BaseModel):
    cement: float
    blast_furnace_slag: float
    fly_ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age: float

    class Config:
        json_schema_extra = {
            "example": {
                "cement": 540.0,
                "blast_furnace_slag": 0.0,
                "fly_ash": 0.0,
                "water": 162.0,
                "superplasticizer": 2.5,
                "coarse_aggregate": 1040.0,
                "fine_aggregate": 676.0,
                "age": 28.0
            }
        }

# --- Route de base (Health check) ---
@app.get("/")
def home():
    return {"message": "API Concrete Strength is running properly 🚀"}





# --- Route de prédiction ---
@app.post("/predict")
def predict_strength(data: ConcreteInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Convertir les données reçues en DataFrame
        # L'ordre des colonnes doit être STRICTEMENT le même que lors de l'entraînement
        input_data = pd.DataFrame([[
            data.cement,
            data.blast_furnace_slag,
            data.fly_ash,
            data.water,
            data.superplasticizer,
            data.coarse_aggregate,
            data.fine_aggregate,
            data.age
        ]], columns=[
            'cement', 'blast_furnace_slag', 'fly_ash', 'water', 
            'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'
        ])

        # 2. Appliquer le scaling (Normalisation)
        # Si tu n'as pas de scaler, commente ces lignes :
        if scaler:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data

        # 3. Prédiction
        prediction = model.predict(input_scaled)

        # 4. Retourner le résultat
        return {
            "strength_prediction": float(prediction[0]),
            "unit": "MPa"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
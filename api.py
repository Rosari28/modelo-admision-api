from fastapi import FastAPI
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("modelo_admision.pkl")

# Inicializar FastAPI
app = FastAPI()

# Definir la ruta de predicción
@app.post("/predecir")
def predecir(data: dict):
    df = pd.DataFrame([data])
    probabilidad = model.predict_proba(df)[:, 1][0]  # Obtener probabilidad
    return {"probabilidad_admision": probabilidad}

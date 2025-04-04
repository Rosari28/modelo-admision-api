from fastapi import FastAPI
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("modelo_admision.pkl")

# Inicializar FastAPI
app = FastAPI()

# Ruta base para verificar que la API funciona
@app.get("/")
def home():
    return {"mensaje": "La API está activa. Usa /predecir para hacer una predicción."}

# Ruta de predicción
@app.post("/predecir")
def predecir(data: dict):
    df = pd.DataFrame([data])
    probabilidad = model.predict_proba(df)[:, 1][0]
    return {"probabilidad_admision": probabilidad}

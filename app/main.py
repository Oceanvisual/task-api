from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from app.schemas.predict import PredictRequest, PredictResponse

MODEL_PATH = Path("models/wine_model.pkl")
CLASS_NAMES = ["class_0", "class_1", "class_2"]
FEATURE_ORDER = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Грузим модель один раз при старте
    app.state.model = joblib.load(MODEL_PATH)
    yield
    # Тут можно было бы закрыть коннекшены к БД, но у нас их нет


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    model = request.app.state.model

    # Превращаем Pydantic-объект в DataFrame с теми же именами колонок,
    # которые модель видела на обучении. by_alias=True вернёт
    # оригинальное имя od280/od315_of_diluted_wines со слэшем.
    row = pd.DataFrame([req.model_dump(by_alias=True)])
    row = row[FEATURE_ORDER]

    proba = model.predict_proba(row)[0]
    pred_idx = int(proba.argmax())

    return PredictResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        probabilities={CLASS_NAMES[i]: float(round(p, 4)) for i, p in enumerate(proba)},
    )
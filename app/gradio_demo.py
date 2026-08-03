from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = Path("models/wine_model.pkl")
CLASS_NAMES = ["class_0", "class_1", "class_2"]
FEATURE_ORDER = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline",
]

model = joblib.load(MODEL_PATH)


def predict_wine(
    alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
    total_phenols, flavanoids, nonflavanoid_phenols,
    proanthocyanins, color_intensity, hue,
    od280_od315_of_diluted_wines, proline,
):
    row = pd.DataFrame([{
        "alcohol": alcohol, "malic_acid": malic_acid, "ash": ash,
        "alcalinity_of_ash": alcalinity_of_ash, "magnesium": magnesium,
        "total_phenols": total_phenols, "flavanoids": flavanoids,
        "nonflavanoid_phenols": nonflavanoid_phenols,
        "proanthocyanins": proanthocyanins, "color_intensity": color_intensity,
        "hue": hue,
        "od280/od315_of_diluted_wines": od280_od315_of_diluted_wines,
        "proline": proline,
    }])[FEATURE_ORDER]
    proba = model.predict_proba(row)[0]
    return {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}


demo = gr.Interface(
    fn=predict_wine,
    inputs=[
        gr.Number(label="alcohol", value=13.5),
        gr.Number(label="malic_acid", value=1.8),
        gr.Number(label="ash", value=2.4),
        gr.Number(label="alcalinity_of_ash", value=18.0),
        gr.Number(label="magnesium", value=105.0),
        gr.Number(label="total_phenols", value=2.7),
        gr.Number(label="flavanoids", value=2.9),
        gr.Number(label="nonflavanoid_phenols", value=0.28),
        gr.Number(label="proanthocyanins", value=1.85),
        gr.Number(label="color_intensity", value=5.5),
        gr.Number(label="hue", value=1.05),
        gr.Number(label="od280/od315_of_diluted_wines", value=3.2),
        gr.Number(label="proline", value=1180.0),
    ],
    outputs=gr.Label(label="Распределение вероятностей по сортам"),
    title="🍷 Wine classifier (minimal demo)",
    description="Введите 13 показателей лабораторного анализа партии",
)

if __name__ == "__main__":
    demo.launch()
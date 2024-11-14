from transformers import pipeline
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SENTIMENT_MODEL = "../models/rubert-tiny2-russian-sentiment"


class Sengine:

    def __init__(self, model_path: str = SENTIMENT_MODEL):
        self.model = pipeline("sentiment-analysis", model=model_path)

    def predict(self, text: str) -> dict:
        return self.model(text)[0]["label"]
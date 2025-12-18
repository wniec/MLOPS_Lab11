import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from tokenizers import Tokenizer
from src.scripts.settings import TOKENIZER_PATH, ONNX_MODEL_PATH, ONNX_CLASSIFIER_PATH
from pydantic import BaseModel

app = FastAPI()

tokenizer = Tokenizer.from_file("."+TOKENIZER_PATH+"/tokenizer.json")

embedding_session = ort.InferenceSession("."+ONNX_MODEL_PATH)

classifier_session = ort.InferenceSession("."+ONNX_CLASSIFIER_PATH)

SENTIMENT_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        cleaned_text = request.text
        encoded = tokenizer.encode(cleaned_text)

        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])

        embedding_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        embeddings = embedding_session.run(None, embedding_inputs)[0]

        classifier_input_name = classifier_session.get_inputs()[0].name

        classifier_inputs = {
            classifier_input_name: embeddings.astype(np.float32)
        }
        print("got here")
        print(classifier_inputs)

        prediction_output = classifier_session.run(None, classifier_inputs)[0]

        predicted_index = np.argmax(prediction_output[0])

        label = SENTIMENT_MAP.get(int(predicted_index), "unknown")

        return {
            "label": label,
            "confidence_score": float(np.max(prediction_output[0]))  # Optional: return raw score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(f"Internal server error: {e}"))

if __name__ == "__main__":
    uvicorn.run(app)

# export_classifier_to_onnx.py

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from settings import ONNX_CLASSIFIER_PATH, CLASSIFIER_PATH, EMBEDDING_DIM

from onnx import save


def export_classifier_to_onnx():
    print(f"Loading classifier from {CLASSIFIER_PATH}...")
    classifier = joblib.load("./"+CLASSIFIER_PATH)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, EMBEDDING_DIM]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(classifier, initial_types=initial_type)

    print(f"Saving ONNX model to {ONNX_CLASSIFIER_PATH}...")

    onnx.save(onnx_model, ONNX_CLASSIFIER_PATH)

export_classifier_to_onnx()
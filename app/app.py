import onnxruntime as ort
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import string
import regex as re
from pyvi import ViTokenizer
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load stopwords list
stw_list = []
with open('./model/vietnamese-stopwords-dash.txt', 'r', encoding='utf8') as f:
    stw_list = [word.strip() for word in f.read().split('\n') if word.strip()]

# Load Word2Vec model
with open('./model/vocab.json', 'r') as f:
    vocab = json.load(f)

vectors = np.load('./model/m240107.model.vi.bin.wv.vectors.npy')
vectors_lockf = np.load('./model/m240107_lockf.npy')

w2v_model = KeyedVectors(vector_size=vectors.shape[1])
w2v_model.add_vectors(list(vocab.keys()), vectors)
w2v_model.vectors_lockf = vectors_lockf

onnx_session = ort.InferenceSession("./model/svm_model.onnx")

app = FastAPI()

class TextInput(BaseModel):
    text: str

def preprocess_and_filter(text):
    text = re.sub(rf'[{string.punctuation}\d\n]', '', text)
    tokens = [w.strip().lower() for w in text.split() if w]
    text = ' '.join(tokens)
    tokens = ViTokenizer.tokenize(text).split()
    tokens = [tk for tk in tokens if tk not in stw_list]
    word_freq = Counter(tokens)
    filtered_tokens = [word for word in tokens if word_freq[word] >= 1]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

def transform_text(text):
    tokens = text.split()
    word_vectors = [w2v_model[word] for word in tokens if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(w2v_model.vector_size)

@app.post("/classify")
async def classify_text(input_data: TextInput):
    text = input_data.text
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    
    processed_text = preprocess_and_filter(text)
    features = transform_text(processed_text).reshape(1, -1)
    
    ort_inputs = {onnx_session.get_inputs()[0].name: features.astype(np.float32)}
    
    try:
        prediction = onnx_session.run(None, ort_inputs)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    logging.info(f"Received text: {text}")
    logging.info(f"Processed text: {processed_text}")
    logging.info(f"Prediction: {prediction[0]}")
    
    return {"text": text, "processed_text": processed_text, "prediction": int(prediction[0])}

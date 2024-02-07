from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sklearn.svm import SVC
import pickle
import numpy as np
import uvicorn
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load the trained model
def load_model():
    model_filename = 'svc_model.pkl'
    vectorizer_filename = 'tfidf_vectorizer.pkl'

    with open(model_filename,'rb') as f:
        loaded_classifier = pickle.load(f)
    
    with open(vectorizer_filename,'rb') as f:
        vector = pickle.load(f)
    # vector = TfidfVectorizer()

    yield loaded_classifier, vector

class InputData(BaseModel):
    sentence: str

@app.post("/predict")
def predict(data: InputData, model:Tuple[SVC, TfidfVectorizer] = Depends(load_model) ):
    try:
        print("input",data.sentence)
        svc, vector = model
        x = vector.transform([data.sentence])
        prediction = svc.predict(x)
        print("output",prediction.item())
        return {"prediction": prediction.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8010)
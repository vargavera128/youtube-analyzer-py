from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from typing import List

app = FastAPI()

classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

class CommentRequest(BaseModel):
    comments: List[str]

@app.post("/analyze")

def analyze(request: CommentRequest):
    results = classifier(request.comments)
    sentiments = []
    for r in results:
        
        stars = int(r["label"][0]) 

        if stars <= 2:
            sentiments.append("negatív")

        elif stars == 3:
            sentiments.append("semleges")

        else:
            sentiments.append("pozitív")

    return {"results": sentiments}

from fastapi import FastAPI
from pydantic import BaseModel
from app.model.request_response import gen_answer


app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    response: dict


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    response = gen_answer(payload.text)
    answer  = response["answer"]
    return {"response": response,
            "answer": answer}  
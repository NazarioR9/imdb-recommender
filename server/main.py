from fastapi import FastAPI
from .api import *

app = FastAPI()
recommender = Recommender()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/recommend")
def forecast(item: Item):
    outputs = recommender(item)

    return outputs

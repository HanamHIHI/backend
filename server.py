# API
from fastapi import FastAPI
from pydantic import BaseModel

import recommender_in_10min

# Create the FastAPI app
app = FastAPI()

class Text(BaseModel):
    text: str

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post('/predict/')
async def create_item(text: Text):
    return recommender_in_10min.predict(text.text)
# API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import recommender_in_10min

# Create the FastAPI app
app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

class Text(BaseModel):
    text: str

@app.get("/")
def root():
    return {"Hello": "World"}

@app.options("/")
def response_options():
    return {"Hallo": "Wollo"}

@app.post('/predict/')
async def create_item(text: Text):
    return recommender_in_10min.predict(text.text)